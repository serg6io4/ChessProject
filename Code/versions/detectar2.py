import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\tablero2.jpg')

grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar detección de bordes con Canny
bordes = cv2.Canny(imagen, 100, 200)

# Aplicar la Transformada de Hough para detección de líneas
lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Seleccionar las líneas con tamaño similar(debido a que como es un cuadrado tiene que tener esquinas similares)
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
    
    # Obtengo las coordenadas de las líneas resultantes
    coordenadas_x = [x for linea in lineas_seleccionadas for x, _, _, _ in linea]
    coordenadas_y = [y for linea in lineas_seleccionadas for _, y, _, _ in linea]
    min_x = min(coordenadas_x)
    max_x = max(coordenadas_x)
    min_y = min(coordenadas_y)
    max_y = max(coordenadas_y)
    
    # Recorto la región que me interesa de la imagen original, según las coordenadas de esas líneas
    imagen_recortada = imagen[min_y:max_y, min_x:max_x]
    
    # Mostrar la imagen recortada resultante
    cv2.imshow('Imagen recortada', imagen_recortada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
