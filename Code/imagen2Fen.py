import cv2
import numpy as np
from detectarTableros import recortarTableros
from recortarTablero import recortar
from recortarfoto2 import recortarFoto
def obtener_coordenadas(event, x, y, flags, param):
    #Para obtener las coordenadas haciendo click en las diferentes esquinas del tablero
    global contador_clics, coordenadas
    if event == cv2.EVENT_LBUTTONDOWN:
        coordenadas.append((x, y))
        contador_clics += 1
        if contador_clics == 4:
            cv2.destroyWindow('Imagen')

def ordenar_puntos(puntos):
    #Esto es para ordenar los puntos(aunque de una manera bruta)
    puntos = np.array(puntos, dtype=np.float32)
    suma_puntos = puntos.sum(axis=1)
    diferencia_puntos = np.diff(puntos, axis=1)
    
    puntos_ordenados = np.zeros((4, 2), dtype=np.float32)
    puntos_ordenados[0] = puntos[np.argmin(suma_puntos)]
    puntos_ordenados[1] = puntos[np.argmin(diferencia_puntos)]
    puntos_ordenados[2] = puntos[np.argmax(suma_puntos)]
    puntos_ordenados[3] = puntos[np.argmax(diferencia_puntos)]
    
    return puntos_ordenados

def aplicar_transformacion(imagen, coordenadas, ancho, alto):
    #Obtengo los puntos de coordenadas de la imagen que he seleccionado
    puntos_origen = ordenar_puntos(coordenadas)
    # Defino las coordenadas de destino para la transformación
    puntos_destino = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
    # Calculo la matriz de transformación perspectiva
    matrix_transformacion = cv2.getPerspectiveTransform(puntos_origen, puntos_destino)
    # Aplico la transformación perspectiva a la imagen original
    imagen_transformada = cv2.warpPerspective(imagen, matrix_transformacion, (ancho, alto))
    return imagen_transformada

def procesar_imagen(ruta_imagen):
    global coordenadas, contador_clics

    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.resize(imagen, (600, 600))
    # Variables para almacenar las coordenadas y el contador de clics
    coordenadas = []
    contador_clics = 0
    #Me creo una ventana sobre la que visualizaremos y trabajemos
    cv2.namedWindow('Imagen')
    cv2.setMouseCallback('Imagen', obtener_coordenadas)

    while True:
        #Mostrar los puntos
        imagen_mostrada = imagen.copy()
        for punto in coordenadas:
            cv2.circle(imagen_mostrada, punto, 5, (0, 255, 0), -1)
        cv2.imshow('Imagen', imagen_mostrada)

        #Para que no se mantenga la imagen en la que ya se ha extraido las coordenadas
        if contador_clics == 4:
            cv2.destroyWindow('Imagen')
            break

        if cv2.waitKey(1) == 27:
            break
    #Realizamos la operación y devolvemos la imagen resultante
    imagen_transformada = aplicar_transformacion(imagen, ordenar_puntos(coordenadas), 600, 600)
    return imagen_transformada

#Cargamos la ruta de la imagen y se la pasamos a procesar
ruta_imagen = 'C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\foto7.jpg'
esFoto = True
imagen_procesada = procesar_imagen(ruta_imagen)
if esFoto:
    recortarFoto(imagen_procesada, 20)
    #recortar(imagen_recortada)
else:
    imagen_recortada = recortarTableros(imagen_procesada)
    recortar(imagen_recortada)





