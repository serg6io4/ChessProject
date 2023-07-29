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
from shapely.geometry import Polygon #Esto es para IOU

### IoU !!!!!!!!!!!!!!!!!!
def poligono(ruta_txt):
   with open(ruta_txt, 'r') as gt_f:
    gt_array = gt_f.readlines()
    gt_array = [x.split() for x in gt_array]
    gt_array = np.array(gt_array).astype('int32')
    gt_array = np.array(gt_array).flatten().astype(np.int32)

    # define corners of image patch
    top_left_point     = (gt_array[0], gt_array[1])
    bottom_left_point  = (gt_array[2], gt_array[3])
    bottom_right_point = (gt_array[4], gt_array[5])
    top_right_point    = (gt_array[6], gt_array[7])
    
    Poligono = Polygon([top_left_point, bottom_left_point, bottom_right_point, top_right_point])
    return Poligono

def IOU (Poligono1, Poligono2):
    polygon1 = Poligono1
    polygon2 = Poligono2
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou

ruta1 = "C:\\Users\\sergi\\Desktop\\transform_images\\dataset\\playchess-0010.txt"
ruta2 = "C:\\Users\\sergi\\Desktop\\transform_images\\dataset\\playchess-0010-prediction.txt"
print(IOU(poligono(ruta1), poligono(ruta2)))

