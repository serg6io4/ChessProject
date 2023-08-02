import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def cargar_modelo(ruta_modelo):
    # Cargar el modelo previamente entrenado
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 13)

    # Cargar los pesos del modelo entrenado
    model.load_state_dict(torch.load(ruta_modelo))
    model.eval()
    return model

def predecir(imagen, modelo):
    """
    Clasifica el tipo de imagen mediante un modelo preentrenado y escribe cuál es

    :param imagen: imagen a predecir
    :param model: modelo preentrenado
    :return: caracter de la pieza de ajedrez clasificada
    """
    # Convierte el array NumPy a un objeto de imagen PIL
    imagen_pil = Image.fromarray(imagen)

    # Definir transformación para la imagen de entrada
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Preprocesar la imagen para que coincida con las transformaciones utilizadas durante el entrenamiento
    imagen_preprocesada = transform(imagen_pil)
    imagen_preprocesada = imagen_preprocesada.unsqueeze(0)  # Agregar una dimensión extra para el tamaño del batch

    # Pasar la imagen preprocesada a través del modelo para obtener las predicciones
    with torch.no_grad():
        outputs = modelo(imagen_preprocesada)

    # Interpretar las predicciones para obtener la clase predicha
    _, predicted_class = torch.max(outputs, 1)

    # Tienes un diccionario que mapea el índice de clase a su nombre(pytorch)
    # (por ejemplo, si la clase 0 es "peon", la clase 1 es "torre", etc.)
    # Aquí voy a realizar una modificación a futuro
    clases = {0: "b", 1: "B", 2: "1", 3: "k", 4: "K", 5: "n", 6: "N", 7: "p", 8: "P", 9: "q", 10: "Q", 11: "r", 12: "R"}

    clase_predicha = clases[predicted_class.item()]
    return clase_predicha

def dividir_cada_ocho(string):
    """
    Divide un string cada 8 caracteres, añadiendo "/" después de cada grupo de 8 caracteres.

    :param string: String a dividir
    :return: String con "/" añadido después de cada grupo de 8 caracteres
    """
    return '/'.join(string[i:i+8] for i in range(0, len(string), 8))

def convertir_a_FEN(tablero):
    """
    Obtiene una versión sin rectificar de la notación FEN y lo rectifica

    :param string: String a rectificar
    :return: String rectificado con la notación FEN
    """
    tablero = tablero.split('/')
    nueva_FEN = ''
    for fila in tablero:
        nueva_fila = ''
        count = 0
        for c in fila:
            if c.isdigit():
                count += int(c)
            else:
                if count > 0:
                    nueva_fila += str(count)
                    count = 0
                nueva_fila += c
        if count > 0:
            nueva_fila += str(count)
        nueva_FEN += nueva_fila + '/'
    nueva_FEN = nueva_FEN[:-1]  # Eliminar la última barra '/'
    return nueva_FEN


def recortar(imagen, modelo):
    """
    Obtención de las casillas establecidas por el ancho y alto de una imagen

    :param imagen: Imagen del tablero de ajedrez
    :param modelo: Modelo cargado previamente
    :return: Lista de FEN representando las casillas del tablero
    """
    # Obtener las dimensiones de la imagen
    ancho, alto = imagen.shape[:2]

    # Calcular el tamaño de cada casilla (dividir entre 8, ya que el tablero tiene que ser casi perfecto linealmente)
    tamaño_casilla_x = ancho // 8
    tamaño_casilla_y = alto // 8

    tamaño_casilla_x = int(tamaño_casilla_x)
    tamaño_casilla_y = int(tamaño_casilla_y)

    kernel = np.array([[-1, -1, -1], [-1, 1.5 + 8, -1], [-1, -1, -1]])
    
    fenNotation = ""  # Lista para almacenar las notaciones FEN

    for fila in range(8):
        for columna in range(8):
            # Aquí obtengo las casillas según las coordenadas
            x1 = columna * tamaño_casilla_x
            y1 = fila * tamaño_casilla_y
            x2 = x1 + tamaño_casilla_x
            y2 = y1 + tamaño_casilla_y

            # Aquí recorto la casilla del punto actual
            casilla = imagen[y1:y2, x1:x2]
            #cv2.imshow("casilla",casilla)
            #cv2.waitKey(0)
            casilla = cv2.filter2D(casilla, -1, kernel)
            casilla = cv2.convertScaleAbs(casilla, alpha=0.7, beta=0)
            clase_predicha = predecir(casilla, modelo)
            #print(clase_predicha)
            # Agrego la clase predicha (notación FEN) a la lista de FEN
            fenNotation = fenNotation + clase_predicha

    fenNotation = dividir_cada_ocho(fenNotation)
    fenNotation = convertir_a_FEN(fenNotation)
    return fenNotation


def sumar_1_entre_piezas(cadena):
    resultado = ""
    suma_temporal = 0

    for caracter in cadena:
        if caracter.isdigit():
            suma_temporal = suma_temporal * 10 + int(caracter)
        else:
            if suma_temporal > 0:
                resultado += str(suma_temporal)
                suma_temporal = 0
            resultado += caracter

    if suma_temporal > 0:
        resultado += str(suma_temporal)

    resultado = resultado.replace('1', '')  # Eliminamos todos los dígitos '1' restantes

    return resultado



