import cv2
import numpy as np
from recortarTablero import recortar
from detectline import recortar_pre

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
    return imagen_transformada, matrix_transformacion

def procesar_imagen(ruta_imagen):
    global coordenadas, contador_clics

    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    alto, ancho = imagen.shape[:2]
    
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
    coordenadas_ordenadas = ordenar_puntos(coordenadas)
    imagen_transformada, matrix_transformacion = aplicar_transformacion(imagen, coordenadas_ordenadas, alto, ancho)
    return imagen_transformada, coordenadas_ordenadas, matrix_transformacion

def calcular_coordenadas_finales(coordenadas_recortadas, ancho_original, alto_original, matriz_transformacion):
    # Obtiene las dimensiones de la imagen recortada
    ancho_recortado, alto_recortado = coordenadas_recortadas[1][0], coordenadas_recortadas[2][1]

    # Calcula la escala en x e y (debido a que esto sería como el offset)
    escala_x = ancho_original / ancho_recortado
    escala_y = alto_original / alto_recortado

    # Calcula las coordenadas finales en la imagen recortada(simplemente aplicamos la escala a los puntos obtenidos)
    coordenadas_finales_recortadas = []
    for punto in coordenadas_recortadas:
        x = int(punto[0] * escala_x)
        y = int(punto[1] * escala_y)
        coordenadas_finales_recortadas.append((x, y))

    # Aplica la transformación perspectiva inversa a las coordenadas recortadas(aplicamos la destransformacion)
    coordenadas_finales_originales = []
    for punto in coordenadas_finales_recortadas:
        punto_homogeneo = np.array([[punto[0]], [punto[1]], [1]])
        punto_transformado = np.dot(np.linalg.inv(matriz_transformacion), punto_homogeneo)
        punto_transformado /= punto_transformado[2]
        x = int(punto_transformado[0])
        y = int(punto_transformado[1])
        coordenadas_finales_originales.append((x, y))

    return coordenadas_finales_originales



#Cargamos la ruta de la imagen y se la pasamos a procesar
ruta_imagen = 'C:\\Users\\sergi\\Desktop\\transform_images\\dataset\\chess-0010.png'
#Obtengo la imagen del marco de seleccion, las coordenadas de ese marco y la matrix que se ha aplicado
imagen_selec, coordenadas_originales, matrix= procesar_imagen(ruta_imagen)
imagen_original = cv2.imread(ruta_imagen)
alto, ancho = imagen_original.shape[:2]#Lo necesito para sacar las coordenadas originales
imagen_pre, coordenadas_puntos= recortar_pre(imagen_selec)
if(imagen_pre is not None):
    #Se le tiene que pasar las coordenadas obtenidas, el alto y el ancho de la imagen original y la matrix de transformacion
    print(calcular_coordenadas_finales(coordenadas_puntos,alto, ancho, matrix))
    cv2.imshow("Previsualizacion", imagen_pre)
    cv2.waitKey(0)
    #recortar(imagen_n)
else:
    print("Vuelva a intentarlo")

