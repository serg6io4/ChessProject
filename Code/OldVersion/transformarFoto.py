import cv2
import numpy as np

# Variables para almacenar las coordenadas y el contador de clics
coordenadas = []
contador_clics = 0

# Función de retroalimentación del evento del mouse
def obtener_coordenadas(event, x, y, flags, param):
    global contador_clics
    if event == cv2.EVENT_LBUTTONDOWN:
        coordenadas.append((x, y))
        contador_clics += 1
        if contador_clics == 4:
            cv2.destroyWindow('Imagen')

# Aquí ordeno los puntos porque en el caso que le pase valores intercambiados, la transformación no se puede llevar a cabo
def ordenar_puntos(puntos):
    puntos = np.array(puntos, dtype=np.float32)
    suma_puntos = puntos.sum(axis=1)
    diferencia_puntos = np.diff(puntos, axis=1)
    
    puntos_ordenados = np.zeros((4, 2), dtype=np.float32)
    puntos_ordenados[0] = puntos[np.argmin(suma_puntos)]
    puntos_ordenados[1] = puntos[np.argmin(diferencia_puntos)]
    puntos_ordenados[2] = puntos[np.argmax(suma_puntos)]
    puntos_ordenados[3] = puntos[np.argmax(diferencia_puntos)]
    
    return puntos_ordenados

def aplicar_Transformacion(coordenadas, ancho, alto):
    # Obtengo las coordenadas de los puntos de referencia
    puntos_origen = ordenar_puntos(coordenadas)

    # Defino las coordenadas de destino para la transformación
    puntos_destino = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])

    # Calculo la matriz de transformación perspectiva
    matrix_transformacion = cv2.getPerspectiveTransform(puntos_origen, puntos_destino)

    # Aplico la transformación perspectiva a la imagen original
    imagen_transformada = cv2.warpPerspective(imagen, matrix_transformacion, (ancho, alto))

    return imagen_transformada

# Esto es para cargar la imagen y hacerla de un tamaño menor para poder verla antes de aplicar los puntos
imagen = cv2.imread('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\foto1.jpg')
imagen = cv2.resize(imagen, (600, 600))

cv2.namedWindow('Imagen')
cv2.setMouseCallback('Imagen', obtener_coordenadas)

while True:
    # Esto es para mostrar la imagen mientras se establecen los puntos en la imagen
    imagen_mostrada = imagen.copy()
    for punto in coordenadas:
        cv2.circle(imagen_mostrada, punto, 5, (0, 255, 0), -1)
    cv2.imshow('Imagen', imagen_mostrada)

    # Salgo si he puesto los 4 puntos
    if contador_clics == 4:
        cv2.destroyWindow('Imagen')
        break

    # Salgo si presiono la tecla 'Esc'
    if cv2.waitKey(1) == 27:
        break

imagenTransf = aplicar_Transformacion(ordenar_puntos(coordenadas), 600, 600)

# Muestro la imagen
cv2.imshow('Imagen Transformada', imagenTransf)
cv2.waitKey(0)
cv2.destroyAllWindows()


