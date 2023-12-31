import argparse
import cv2
from PIL import Image
import numpy as np
from detectline import recortar_pre
from predecir2version import cargar_modelo,recortar,predecir

def obtener_coordenadas(event, x, y, flags, param):
    """
    Obtención de coordenadas, por evento de mouse

    :param:  Evento, posiciones(x,y), Flags y Parámetro
    :return: *se almacena en una varible global las coordenadas*
    """
    #Para obtener las coordenadas haciendo click en las diferentes esquinas del tablero
    global contador_clics, coordenadas
    if event == cv2.EVENT_LBUTTONDOWN:
        coordenadas.append((x, y))
        contador_clics += 1
        if contador_clics == 4:
            cv2.destroyWindow('Imagen')

def ordenar_puntos(puntos):
    """
    Agarra una lista de puntos y los devuelve ordenados

    :param:  Coordenadas desordenadas
    :return: Lista de coordenadas ordenadas(esi, esd, eai, ead)
    """
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
    """
    Aplica una transformacion perspectiva

    :param:  Imagen a transformar, coordenadas de la sección a trasnformar, dimensiones de la imagen
    :return: Imagen transformada(plana), matrix de transformacion aplicada 
    """
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
    """
    Aplica un transformación perspectiva a la imagen

    :param:  Ruta de la imagen
    :return: Devuelve imagen transformada(plana), coordenadas de la seleccion de la transformación
             matrix aplicada para la transformación 
    """
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
    """
    Obtener las coordenadas en la imagen original de las coordenadas recortadas en una transformacion concreta

    :param: Coordenadas de la imagen transformada, tamaño original de la imagen(ancho y alto), matriz aplicada en dicha transformacion previa
    :return: Coordenadas en la imagen original
    """
    # Obtengo las dimensiones de la imagen recortada(se que es un poco cortar por lo sano)
    ancho_recortado, alto_recortado = coordenadas_recortadas[1][0], coordenadas_recortadas[2][1]

    # Calculo la escala en x e y (debido a que esto sería como el offset)
    escala_x = ancho_original / ancho_recortado
    escala_y = alto_original / alto_recortado

    # Calculo las coordenadas finales en la imagen recortada(simplemente aplicamos la escala a los puntos obtenidos)
    coordenadas_finales_recortadas = []
    for punto in coordenadas_recortadas:
        x = int(punto[0] * escala_x)
        y = int(punto[1] * escala_y)
        coordenadas_finales_recortadas.append((x, y))

    # Aplico la transformación perspectiva inversa a las coordenadas recortadas(aplicamos la destransformacion)
    coordenadas_finales_originales = []
    for punto in coordenadas_finales_recortadas:
        punto_homogeneo = np.array([[punto[0]], [punto[1]], [1]])
        punto_transformado = np.dot(np.linalg.inv(matriz_transformacion), punto_homogeneo)
        punto_transformado /= punto_transformado[2]
        x = int(punto_transformado[0])
        y = int(punto_transformado[1])
        coordenadas_finales_originales.append((x, y))
    #No es preciso 100 pero es un calculo muy cercano a lo que se obtiene
    return coordenadas_finales_originales

def coordenadas_txt(coordenadas, ruta):
    """
    Guardar en un txt los puntos extraidos de la imagen

    :param:  Coordenadas extraidas, ruta de la imagen
    :return: "Guardará un txt, con el nombre de la ruta de la imagen más -prediction, con las coordenadas"
    """
    coordenadas_nuevas = [coordenadas[0], coordenadas[2], coordenadas[3], coordenadas[1]]
    # Convertir las coordenadas en una cadena con el formato "x y"
    coordenadas_str = " ".join([f"{x} {y}" for x, y in coordenadas_nuevas])

    # Guardar las coordenadas en un archivo de texto
    nombre_archivo = ruta + "-prediction.txt"
    with open(nombre_archivo, "w") as archivo:
        archivo.write(coordenadas_str)

    print(f"Se han guardado las coordenadas en el archivo: {nombre_archivo}")

def resize_image(input_image_path, target_width):
    """
    Modifica el tamaño de una imagen, con un ancho dado y  altura automática, para poder ser utilizada por el programa 

    :param:  ancho deseado, ruta de la imagen
    """
    # Abrir la imagen desde la ruta proporcionada
    image = Image.open(input_image_path)

    # Calcular la nueva altura proporcional manteniendo la relación de aspecto
    aspect_ratio = image.width / image.height
    target_height = int(target_width / aspect_ratio)

    # Reescalar la imagen
    resized_image = image.resize((target_width, target_height), Image.LANCZOS)

    # Guardar la imagen reescalada en la misma ruta (reemplazando la original)
    resized_image.save(input_image_path)


##################################
#    Realización del programa    #
##################################

# Crear un analizador de argumentos
parser = argparse.ArgumentParser(description='Procesamiento de imágenes y detección de piezas de ajedrez')

# Agregar un argumento para la ruta de la imagen
parser.add_argument('imagen', type=str, help='Ruta de la imagen a procesar')

# Analizar los argumentos de línea de comandos
args = parser.parse_args()

#Cargamos la ruta de la imagen y se la pasamos a procesar
ruta_carpeta = "transform_images\demo"
ruta_imagen = "\\" + args.imagen

#Cambiamos el tamaño de la imagen(debido a que es necesario un tamaño visible en la consola)
resize_image(ruta_carpeta+ruta_imagen, 600)

print("1-Select the corners of the chessboard")
print("Note:Click near the corners of the chessboard")
#Obtengo la imagen del marco de seleccion, las coordenadas de ese marco y la matrix que se ha aplicado
imagen_selec, coordenadas_originales, matrix= procesar_imagen(ruta_carpeta + ruta_imagen)

"""
#Está comentado debido a que se utilizó para el testeo del algoritmo de detección
#Lo hago para obtener las medidas, se necesitarán para destransformar la perspectiva inicial
imagen_original = cv2.imread(ruta_carpeta + ruta_imagen + ".png")
alto, ancho = imagen_original.shape[:2]
"""
#Recorto la imagen seleccionada
print("2-Showing the process of detection")
imagen_pre, coordenadas_puntos= recortar_pre(imagen_selec)

if(imagen_pre is not None):
    
    """
    #Está comentado debido a que se utilizó para el testeo del algoritmo de detección
    #Se le tiene que pasar las coordenadas obtenidas, el alto y el ancho de la imagen original y la matrix de transformacion
    coordenadas_reales = calcular_coordenadas_finales(coordenadas_puntos,alto, ancho, matrix)
    coordenadas_txt(coordenadas_reales, (ruta_carpeta + ruta_imagen))
    """

    print("3-Showing the detecting chessboard")
    cv2.imshow("Visualizing chessboard", imagen_pre)
    cv2.waitKey(0)
    #Con esto tipifico la imagen a tamaño estándar
    print("4-Resizing the board to perfect square")
    imagen_pre = cv2.resize(imagen_pre, (600,600))

    #Cargo el modelo dentro de una variable
    print("5-Loading the model to predict the chess pieces")
    ruta_modelo = "AI\modelo\mobilenetv2_chess_classification.pt"
    modelo = cargar_modelo(ruta_modelo)

    print("6-Typing to the FEN code")
    # LLamo a la función recortar para recortar el tablero y escribirlo en notación FEN
    fenNotation = recortar(imagen_pre, modelo)
    #La devuelvo por pantalla
    print("FEN NOTATION:")
    print(fenNotation)
    cv2.destroyAllWindows()
    

else:
    
    print("As it could not be detected semi-automatically, we are going to try the following...")
    #Imagen ha sido nula, en la detección semiautomática y ahora procedemos a darle el tablero exacto
    print("1-Select exactly the corners of the chessboard:")
    imagen_selec1, coordenadas_ord, matrix= procesar_imagen(ruta_carpeta + ruta_imagen)
    cv2.imshow("Visualizing chessboard", imagen_selec1)
    cv2.waitKey(0)

    print("2-Resizing the board to perfect square")
    imagen_selec1 = cv2.resize(imagen_selec1, (600,600))

    #Cargo el modelo dentro de una variable
    print("3-Loading the model to predict the chess pieces")
    ruta_modelo = "AI\modelo\mobilenetv2_chess_classification.pt"
    modelo = cargar_modelo(ruta_modelo)
    
    print("4-Typing to the FEN code")
    # LLamo a la función recortar para recortar el tablero y escribirlo en notación FEN
    fenNotation = recortar(imagen_selec1, modelo)
    #La devuelvo por pantalla
    print("FEN NOTATION:")
    print(fenNotation)
    cv2.destroyAllWindows()

    
    