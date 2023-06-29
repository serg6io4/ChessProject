import cv2
import numpy as np

# Carga de la imagen
image = cv2.imread('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\foto3.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Aplicar el filtrado por desenfoque bilateral
d = 15  # Diámetro del vecindario utilizado para el filtrado bilateral
sigma_color = 100  # Valor de desviación estándar en el espacio de color
sigma_space = 100  # Valor de desviación estándar en el espacio de coordenadas
bilateral_filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Aplicar el filtrado de mediana a la imagen en escala de grises
kernel_size = 3  # Tamaño del kernel de filtrado de mediana
median_filtered_image = cv2.medianBlur(bilateral_filtered_image, kernel_size)

# Disminuir el brillo mediante la reducción del valor de píxeles
#brightness_factor = 1  # Factor de brillo deseado (0.0 a 1.0)
#darkened_image = np.clip(median_filtered_image * brightness_factor, 0, 255).astype(np.uint8)

# Redimensionar la imagen
image = cv2.resize(image, (600, 600))
darkened_image = cv2.resize(median_filtered_image, (600, 600))

edges = cv2.Canny(darkened_image, threshold1=30, threshold2=50)
cv2.imshow("Bordes", edges)
cv2.waitKey(0)
# Aplicar la transformada de Hough para detección de líneas
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=65, minLineLength=50, maxLineGap=7)

# Dibujar las líneas detectadas en la imagen original
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(darkened_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Mostrar la imagen original con las líneas detectadas
cv2.imshow('Imagen original con líneas detectadas', darkened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
