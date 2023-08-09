import cv2
import numpy as np

# Carga de la imagen
image = cv2.imread('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\foto.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Aplicar el filtrado por desenfoque bilateral
d = 15  # Diámetro del vecindario utilizado para el filtrado bilateral
sigma_color = 200  # Valor de desviación estándar en el espacio de color
sigma_space = 200  # Valor de desviación estándar en el espacio de coordenadas
bilateral_filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Aplicar el filtrado de mediana a la imagen en escala de grises
kernel_size = 5  # Tamaño del kernel de filtrado de mediana
median_filtered_image = cv2.medianBlur(bilateral_filtered_image, kernel_size)

# Disminuir el brillo mediante la reducción del valor de píxeles
brightness_factor = 0.3  # Factor de brillo deseado (0.0 a 1.0)
darkened_image = np.clip(median_filtered_image * brightness_factor, 0, 255).astype(np.uint8)



image = cv2.resize(image, (600,600))
darkened_image = cv2.resize(darkened_image, (600,600))

# Mostrar la imagen original, la imagen con brillo reducido y el desenfoque promedio
cv2.imshow('Imagen original', image)
cv2.imshow('Todos los filtros y disminución de brillo', darkened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()