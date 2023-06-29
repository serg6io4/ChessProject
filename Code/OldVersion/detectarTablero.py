import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\Captura2.jpg')

grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar detección de bordes con Canny
bordes = cv2.Canny(grises, 100, 200)

# Aplicar la Transformada de Hough para detección de líneas
lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=200, minLineLength=100, maxLineGap=10)

# Seleccionar las líneas con tamaño similar
if lineas is not None:
    lineas_seleccionadas = []
    for i in range(len(lineas)):
        x1, y1, x2, y2 = lineas[i][0]
        longitud1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        for j in range(i+1, len(lineas)):
            x3, y3, x4, y4 = lineas[j][0]
            longitud2 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
            
            # Comparar las longitudes de las líneas
            diferencia = np.abs(longitud1 - longitud2)
            umbral = 1  # Establecer el umbral de diferencia de longitud
            
            if diferencia < umbral:
                lineas_seleccionadas.append(lineas[i])
                lineas_seleccionadas.append(lineas[j])
    
    # Dibujar las líneas seleccionadas en la imagen original
    for linea in lineas_seleccionadas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Mostrar la imagen original con las líneas seleccionadas
cv2.imshow('Imagen con líneas seleccionadas', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
