import cv2
import numpy as np
from predecir2version import predecir

def recortar(imagen):
    """
    Obtención de las casillas establecidas por el ancho y alto de una imagen

    :param imagen: Imagen del tablero de ajedrez
    :return: Array de imágenes representando las casillas del tablero(0-63)
    """
    # Obtener las dimensiones de la imagen
    fenNotation = []
    ancho, alto = imagen.shape[:2]

    # Calcular el tamaño de cada casilla (dividir entre 8, ya que el tablero tiene que ser casi perfecto linealmente)
    tamaño_casilla_x = ancho // 8
    tamaño_casilla_y = alto // 8

    tamaño_casilla_x = int(tamaño_casilla_x)
    tamaño_casilla_y = int(tamaño_casilla_y)

    casillas_array = []  # Lista para almacenar las casillas (imágenes)

    for fila in range(8):
        for columna in range(8):
            # Aquí obtengo las casillas según las coordenadas
            x1 = columna * tamaño_casilla_x
            y1 = fila * tamaño_casilla_y
            x2 = x1 + tamaño_casilla_x
            y2 = y1 + tamaño_casilla_y

            # Aquí recorto la casilla del punto actual
            casilla = imagen[y1:y2, x1:x2]
            fenNotation.append(predecir(casilla))

    return fenNotation

