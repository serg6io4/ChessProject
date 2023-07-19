import numpy as np
import cv2

def canny(image):
    """
    Aplicación de detección de bordes por algoritmo Canny(umbrales automáticos)

    :param:  imagen 
    :return: imagen con bordes detectados
    """
    sigma = 0.25
    v = np.median(image)
    img = cv2.medianBlur(image, 3)
    img = cv2.GaussianBlur(image, (3, 3), 0)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(img, lower, upper)

def lineas(canny_image, original_image, rho=1, theta=np.pi/360, threshold=200):
    """
    Búsqueda de las líneas de la imagen con bordes detectados, 
    Búsqueda de puntos de intersecciones entre líneas

    :param:  imagen con bordes detectados, imagen original, rho, theta, umbral
    :return: Una lista de los puntos de intersección entre las líneas
    """
    # Aplicar la transformada de Hough a la imagen Canny
    lines = cv2.HoughLines(canny_image, rho, theta, threshold)

    # Encontrar los puntos de intersección
    def encontrar_puntos_interseccion(lines):
        puntos_interseccion = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                a1 = np.cos(theta1)
                b1 = np.sin(theta1)
                x1 = a1 * rho1
                y1 = b1 * rho1
                a2 = np.cos(theta2)
                b2 = np.sin(theta2)
                x2 = a2 * rho2
                y2 = b2 * rho2

                # Calcular el punto de intersección
                determinante = a1 * b2 - a2 * b1
                if determinante != 0:
                    punto_x = (b2 * x1 - b1 * x2) / determinante
                    punto_y = (-a2 * y1 + a1 * y2) / determinante
                    puntos_interseccion.append((int(punto_x), int(punto_y)))

        return puntos_interseccion

    puntos_interseccion = encontrar_puntos_interseccion(lines)

    # Dibujar las líneas y los puntos de intersección en la imagen original
    line_image = np.copy(original_image)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    

    # Dibujar los puntos de intersección
    for punto in puntos_interseccion:
        cv2.circle(line_image, punto, 4, (255, 0, 0), -1)

    cv2.imshow("Lineas y puntos de intersección", line_image)
    cv2.waitKey(0)
    return puntos_interseccion

def punto_esquina(x, y, puntos_interseccion, esquina_x, esquina_y):
    """
    Búsqueda del punto más céntrico encontrado de una lista de puntos sobre
    una zona concreta 

    :param:  Las posiciones de las esquinas de la imagen, la lista de puntos, el tamaño de la zona en x e y
    :return: Punto más céntrico en la zona delimitada
    """
    mejor_punto = None
    mejor_distancia = float('inf')

    for punto in puntos_interseccion:
        punto_x, punto_y = punto
        if x <= punto_x < x + esquina_x and y <= punto_y < y + esquina_y:
            centro_x = punto_x + esquina_x // 2
            centro_y = punto_y + esquina_y // 2
            distancia = np.sqrt((centro_x - punto_x) ** 2 + (centro_y - punto_y) ** 2)
            if distancia < mejor_distancia:
                mejor_punto = punto
                mejor_distancia = distancia

    return mejor_punto

def recuperar_punto(puntos, alto_imagen, ancho_imagen):
    """
    Recuperar las coordenadas faltantes de la búsqueda obtenida a posteriori de los puntos
    en las zonas delimitadas

    :param:  Los puntos obtenidos de las zonas delimitantes, las dimensiones de la imagen
    :return: Lista de puntos concretos de la imagen 
    """
    #Guardamos en variables todas los puntos
    punto1 = puntos[0]
    punto2 = puntos[1]
    punto3 = puntos[2]
    punto4 = puntos[3]
    #Miramos cuantos puntos faltantes tenemos
    puntos_validos = [punto for punto in puntos if punto is not None ]
   
    if (len(puntos_validos))<=1:
        #Si faltan 3 puntos, no se puede hacer nada, porque no sería tan exacto realizar las operaciones
        print("No se pudo recuperar ningún punto")
    elif (len(puntos_validos))==2:
        #Si faltan 2 puntos hay 6 combinaciones posibles:
        # [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        if((punto1 is not None)&(punto2 is not None)):
            punto3 = (punto1[0], alto_imagen - punto1[1])
            punto4 = (punto2[0], alto_imagen - punto2[1])
        elif ((punto1 is not None)&(punto3 is not None)):
            punto2 = (ancho_imagen-punto1[0], punto1[1])
            punto4 = (ancho_imagen-punto3[0], punto3[1])
        elif ((punto1 is not None)&(punto4 is not None)):
            punto2 = (punto4[0], punto1[1])
            punto3 = (punto1[0], punto4[1])
        elif ((punto2 is not None)&(punto3 is not None)):
            punto1 = (punto3[0], punto2[1])
            punto4 = (punto2[0], punto3[1])
        elif ((punto2 is not None)&(punto4 is not None)):
            punto1 = (ancho_imagen-punto2[0], punto2[1])
            punto3 = (ancho_imagen-punto4[0], punto4[1])
        elif ((punto3 is not None)&(punto4 is not None)):
            punto1 = (punto3[0], alto_imagen-punto3[1])
            punto2 = (punto4[0], alto_imagen-punto4[1])
    elif (len(puntos_validos))==3:
        #Si tenemos 3 puntos, solo necesitamos hallar 1,
        # 4 posibilidades:[(1,2,3),(1,2,4),(1,3,4),(2,3,4)]
        if(punto1 is None):
           punto1 = (punto3[0], punto2[1])
        elif (punto2 is None):
            punto2 = (punto4[0], punto1[1])
        elif (punto3 is None):
            punto3 = (punto1[0], punto4[1])
        else:
            punto4 = (punto2[0], punto3[1])
    #Los reservo en la variable puntos y los devuelvo
    puntos = []
    puntos.extend([punto1, punto2, punto3, punto4])
    return puntos


def recortar_pre(imagen):
    """
    Recortar la zona detectada de una imagen pasada, aplicando las funciones anteriores

    :param:  Una imagen
    :return: Una imagen tratada
    """

    #Realizamos Canny para detección de bordes dentro de la imagen que nos pasan
    Canny = canny(imagen)
    #Lo necesitaremos para recuperar las coordenadas que nos falten a posteriori de detectar los puntos
    alto_imagen, ancho_imagen = imagen.shape[:2]
    #Obtenemos los puntos
    puntos = lineas(Canny,imagen)
    #Especificamos la zona a buscar
    cuadrado_tam=40
    #Buscamos en cada una delas zonas cercanas a las esquinas de la imagen
    punto1 = punto_esquina(0, 0, puntos, cuadrado_tam, cuadrado_tam)
    punto2 = punto_esquina(ancho_imagen - cuadrado_tam, 0, puntos, cuadrado_tam, cuadrado_tam)
    punto3 = punto_esquina(0, alto_imagen - cuadrado_tam, puntos, cuadrado_tam, cuadrado_tam)
    punto4 = punto_esquina(ancho_imagen - cuadrado_tam, alto_imagen - cuadrado_tam, puntos, cuadrado_tam, cuadrado_tam)
    
    #Debido a que no siempre va a estar el punto 1 y el 4, tenemos que realizar un sistema el cual pueda recrear los puntos faltantes
    #Para ello será necesario, realizar operaciones con los sistemas de coordenadas con los puntos que tenemos
    puntos = []
    puntos.extend([punto1, punto2, punto3, punto4])
    #Utilizo esto por si tengo 1 o 2 puntos faltantes
    #Reutilizamos variable
    puntos = recuperar_punto(puntos, alto_imagen, ancho_imagen)
    #Cortamos la imagen
    imagen_n = imagen_n = imagen[puntos[0][1]:puntos[3][1], puntos[0][0]:puntos[3][0]]

    return imagen_n, puntos



