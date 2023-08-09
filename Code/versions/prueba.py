import cv2
import numpy as np

def recortar_cuadrado(imagen, coordenadas):

    # Extraer las coordenadas
    x1, y1, x2, y2 = coordenadas
    
    # Recortar el cuadrado de la imagen
    cuadrado_recortado = imagen[y1:y2, x1:x2]
    
    # Mostrar la imagen recortada
    cv2.imshow("Cuadrado Recortado", cuadrado_recortado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cuadrado_recortado

def detectar_lineas(imagen):
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el detector de bordes Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Aplicar la transformada de Hough para detectar líneas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=1, maxLineGap=10)
    
    # Dibujar las líneas detectadas en la imagen original
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Mostrar la imagen con las líneas detectadas
    cv2.imshow("Líneas Detectadas", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen
imagen = cv2.imread('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\tablero3.jpg')
imagen = cv2.resize(imagen, (600,600))
alto, ancho= imagen.shape[:2]
cuadrado_tam=50
# Coordenadas del cuadrado a recortar
coordenadas1 = [0, 0, cuadrado_tam, cuadrado_tam]
coordenadas2 = [ancho - cuadrado_tam, 0, ancho, cuadrado_tam]
coordenadas3 = [0, alto - cuadrado_tam, cuadrado_tam, alto]
coordenadas4 = [ancho - cuadrado_tam, alto - cuadrado_tam, ancho, alto]
# Llamar a la función para recortar y mostrar el cuadrado
cuadrado1 = recortar_cuadrado(imagen, coordenadas1)
cuadrado2 = recortar_cuadrado(imagen, coordenadas2)
cuadrado3 = recortar_cuadrado(imagen, coordenadas3)
cuadrado4 = recortar_cuadrado(imagen, coordenadas4)
detectar_lineas(cuadrado1)
detectar_lineas(cuadrado2)
detectar_lineas(cuadrado3)
detectar_lineas(cuadrado4)







