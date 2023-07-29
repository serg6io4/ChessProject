import os, pdb, shutil, argparse
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from matplotlib import pyplot as plt
from skimage import io
import time
import tensorflow as tf
from PIL import Image
from openpyxl import Workbook, load_workbook
from shapely.geometry import Polygon #Esto es para IOU
# uncomment this line to test this code
#random.seed(0)

SCALE_H = False
# scale_H: The indices of the grid of the target output is
# scaled to [-1, 1]. Set False to stay in normal mode

rho = 100 # The maximum value of pertubation


def _meshgrid(height, width, scale_H = SCALE_H):
    if scale_H:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                        np.linspace(-1, 1, height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    else:
        x_t, y_t = np.meshgrid(range(0,width), range(0,height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    return grid


def _interpolate(im, x, y, out_size, scale_H = SCALE_H):
    # constants
    height = im.shape[0]
    width =  im.shape[1]

    height_f = float(height)
    width_f =  float(width)
    out_height = out_size[0]
    out_width = out_size[1]
    zero = np.zeros([], dtype='int32')
    max_y = im.shape[0] - 1
    max_x = im.shape[1] - 1

    if scale_H:
        # # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

    # do sampling
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Limit the size of the output image
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)

    Ia = im[ y0, x0, ... ]
    Ib = im[ y1, x0, ... ]
    Ic = im[ y0, x1, ... ]
    Id = im[ y1, x1, ... ]

    wa = (x1 -x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    # print 'wabcd...', wa,wb, wc,wd

    # Handle multi channel image
    if im.ndim == 3:
      num_channels = im.shape[2]
      wa = np.expand_dims(wa, 1)
      wb = np.expand_dims(wb, 1)
      wc = np.expand_dims(wc, 1)
      wd = np.expand_dims(wd, 1)
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    # print '--shape of out:', out.shape
    return out


def _transform(theta, input_dim, out_size):
    height, width = input_dim.shape[0], input_dim.shape[1]
    theta = np.reshape(theta, (3, 3))
    # print '--Theta:', theta
    # print '-- Theta shape:', theta.shape

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = out_size[0]
    out_width = out_size[1]
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = np.dot(theta, grid)
    x_s = T_g[0,:]
    y_s = T_g[1,:]
    t_s = T_g[2,:]
    # print '-- T_g:', T_g
    # print '-- x_s:', x_s
    # print '-- y_s:', y_s
    # print '-- t_s:', t_s

    t_s_flat = np.reshape(t_s, [-1])
    # Ty changed
    # x_s_flat = np.reshape(x_s, [-1])
    # y_s_flat = np.reshape(y_s, [-1])
    x_s_flat = np.reshape(x_s, [-1])/t_s_flat
    y_s_flat = np.reshape(y_s, [-1])/t_s_flat


    input_transformed =  _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

    if input_dim.ndim == 3:
      output = np.reshape(input_transformed, [out_height, out_width, -1])
    else:
      output = np.reshape(input_transformed, [out_height, out_width])

    output = output.astype(np.uint8)
    return output


def numpy_transformer(img, H, out_size, scale_H = SCALE_H):
    h, w = img.shape[0], img.shape[1]
    # Matrix M
    M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)

    if scale_H:
        H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
        # print 'H_transformed:', H_transformed
        img2 = _transform(H_transformed, img, [h,w])
    else:
        img2 = _transform(np.linalg.inv(H), img, [h,w])
    return img2


def show_image(location, title, img, width=None):
    if width is not None:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if width is not None:
        plt.show()
        plt.close()




raw_image_path = "C:\\Users\\sergi\\Desktop\\transform_images\\dataset\\"
raw_image_name = "lichess-0002"
 #Cargo el xlsx que es el excel
wb = load_workbook('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Test\\Testeo3.xlsx')
ws = wb.active
ws.title = 'Testeo'

color_image = cv2.imread(raw_image_path + raw_image_name + ".png")

height = color_image.shape[0]
width = color_image.shape[1]


# read corners of image patch
with open(raw_image_path + raw_image_name + ".txt", 'r') as gt_f:
    gt_array = gt_f.readlines()
gt_array = [x.split() for x in gt_array]
gt_array = np.array(gt_array).astype('int32')
gt_array = np.array(gt_array).flatten().astype(np.int32)

# define corners of image patch
top_left_point     = (gt_array[0], gt_array[1])
bottom_left_point  = (gt_array[2], gt_array[3])
bottom_right_point = (gt_array[4], gt_array[5])
top_right_point    = (gt_array[6], gt_array[7])

four_points        = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]


found = False # we generate images until the generated image has the four points inside
while not found:
    perturbed_four_points = []
    for point in four_points:
      perturbed_point = (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho))
      perturbed_four_points.append(perturbed_point)

    # compute Homography
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    try:
      H_inverse = inv(H)
    except:
      print("singular Error!")
    inv_warped_color_image = numpy_transformer(color_image, H_inverse, (width,height))

    # check if points are inside the generated image
    # so that, coordinate points has to be 0 < pixel < width|height
    points_I2 = cv2.perspectiveTransform(np.array([four_points],dtype=np.float32), H_inverse)
    points_I2 = points_I2[0]
    points_I2 = points_I2.astype(int)
    print(points_I2)
    coordinates_I2_outside = []
    for point in points_I2:
      point_outside = [0, 0]
      if point[0]<0 or point[0]>width:
        point_outside[0] = 1
      if point[1]<0 or point[1]>height:
        point_outside[1] = 1
      coordinates_I2_outside.append(point_outside)
    print(coordinates_I2_outside)
    # Check if element 1 (coordinate outside of image) exists in the list
    for sublist in coordinates_I2_outside:
        if 1 in sublist: # print("Element exists"); generate other image
            break
    else: # print("Element does not exist"); we have found an image with points inside
        found = True

# show generated images
top_left_point_I1 = np.array(four_points[0])
top_left_point_I2 = cv2.perspectiveTransform(np.array([[top_left_point_I1]],dtype=np.float32), H_inverse)[0,0]

patches_visualization = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
cv2.circle(patches_visualization,tuple(top_left_point_I1), 20, (255,0,0), -1)
show_image((2, 2, 1), "I1", patches_visualization)
# visualize patch on warped image
patch_warped_visualization = cv2.cvtColor(inv_warped_color_image, cv2.COLOR_BGR2RGB)
cv2.circle(patch_warped_visualization, tuple(np.int32(top_left_point_I2)), 20, (255,0,0), -1)
show_image((2, 2, 2), "I2", patch_warped_visualization)

# warped white image
white_img = np.zeros([height,width,3],dtype=np.uint8)
white_img[:] = 255
inv_warped_image_white = numpy_transformer(white_img, H_inverse, (width,height))
I2_to_I1_white = numpy_transformer(inv_warped_image_white, H, (width,height))

# warp image 2 back using inverse of (H_inverse)
I2_to_I1 = numpy_transformer(inv_warped_color_image, H, (width,height))
I2_to_I1 = cv2.cvtColor(I2_to_I1, cv2.COLOR_BGR2RGB)
show_image((2, 2, 3), "I2 warped to I1", I2_to_I1)
show_image((2, 2, 4), "I2 warped to I1 B&W", I2_to_I1_white)
plt.show()

# save warped image without noise
now_time = int(time.time()*1000) #time in milliseconds
image_path_name = raw_image_path + raw_image_name + "-" + str(now_time) + ".png"
ws.append([raw_image_name + "-" + str(now_time) + ".png", raw_image_name])
cv2.imwrite(image_path_name, inv_warped_color_image)

# write warped image round truth
f_gt = open(raw_image_path + raw_image_name + "-" + str(now_time) + ".txt", 'a')
gt = np.subtract(np.array(perturbed_four_points), np.array(four_points)) # Ground truth is delta displacement
gt = np.array(points_I2).flatten() #.astype(np.float32)
np.savetxt(f_gt, [gt], fmt='%i',delimiter= ' ')
f_gt.close()



# change image color
# a pixel can have a value between 0 and 255
min_val=0
max_val=255
# Randomly shift gamma
random_gamma1 = tf.random.uniform([], 0.9, 1.1)
img1_aug = inv_warped_color_image ** random_gamma1




# Randomly shift brightness
random_brightness1 = tf.random.uniform([], 0.75, 1.5)
img1_aug = img1_aug * random_brightness1


# Randomly shift color
random_colors1 = tf.random.uniform([3], 0.9, 1.1)
white = tf.ones([tf.shape(img1_aug)[0], tf.shape(img1_aug)[1]])
color_image_t = tf.stack([white * random_colors1[i] for i in range(3)], axis=2)
img1_aug *= color_image_t



# Saturate
img1_aug = tf.clip_by_value(img1_aug, min_val, max_val)

img1_aug = img1_aug.numpy()

image_path_name = raw_image_path +  raw_image_name + "-" + str(now_time) + "-color.png"
ws.append([raw_image_name + "-" + str(now_time) + "-color.png", raw_image_name, "S", random_gamma1.numpy(), random_brightness1.numpy(), random_colors1.numpy()[0],random_colors1.numpy()[1], random_colors1.numpy()[2], "None"])
cv2.imwrite(image_path_name, img1_aug)


# Adding Gaussian noise
sigma = random.randint(0, 20) # sigma = [0..20]
noise = np.random.normal(0, sigma, img1_aug.shape) # mu=0
img_noise_array = img1_aug + noise
img_noise_array = np.clip(img_noise_array, 0, 255)
#img_noise = Image.fromarray(np.uint8(img_noise_array))


image_path_name = raw_image_path + raw_image_name + "-" + str(now_time) + "-colorgaussian.png"
ws.append([raw_image_name + "-" + str(now_time) + "-colorgaussian.png", raw_image_name, "S", random_gamma1.numpy(), random_brightness1.numpy(),random_colors1.numpy()[0],random_colors1.numpy()[1], random_colors1.numpy()[2], sigma])
cv2.imwrite( image_path_name, img_noise_array)

#Guardo el trabajo
wb.save('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Test\\Testeo3.xlsx')



