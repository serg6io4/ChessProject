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

#Para recortar el tablero de forma precisa
def recortar_pre(imagen):
    """
    Recortar la zona detectada de una imagen pasada, aplicando las funciones anteriores

    :param:  Una imagen
    :return: Una imagen tratada
    """
    Canny = canny(imagen)
    #cv2.imshow("C", Canny)
    #cv2.waitKey(0)
    alto_imagen, ancho_imagen = imagen.shape[:2]
    #No cogemos lineas, por el simple hecho que las lineas el medio para obtener los puntos
    puntos = lineas(Canny,imagen)
    cuadrado_tam=40
    punto1 = punto_esquina(0, 0, puntos, cuadrado_tam, cuadrado_tam)
    punto2 = punto_esquina(ancho_imagen - cuadrado_tam, 0, puntos, cuadrado_tam, cuadrado_tam)
    punto3 = punto_esquina(0, alto_imagen - cuadrado_tam, puntos, cuadrado_tam, cuadrado_tam)
    punto4 = punto_esquina(ancho_imagen - cuadrado_tam, alto_imagen - cuadrado_tam, puntos, cuadrado_tam, cuadrado_tam)
    
    #Debido a que no siempre va a estar el punto 1 y el 4, tenemos que realizar un sistema el cual pueda recrear los puntos faltantes
    #Para ello será necesario, realizar operaciones con los sistemas de coordenadas con los puntos que tenemos
    puntos = []
    puntos.extend([punto1, punto2, punto3, punto4])
    #Esto lo realizo para saber cuantos puntos me faltan
    puntos_validos = [punto for punto in puntos if punto is not None ]
    imagen_n = None

    if len(puntos_validos)<=1:
        print("No hay suficientes puntos para detectar el tablero")
    elif len(puntos_validos)>=2:
        if((punto1 is not None) & (punto4 is not None)):
            #Realmente solo importa a la hora de recortar estas dos coordenadas
            imagen_n = imagen[punto1[1]:punto4[1], punto1[0]:punto4[0]]

        elif((punto1 is not None) & (punto4 is None)):
            #3 Casos posibles:
            #Tenemos (1,2,3) (1,2), (1,3)
            if((punto2 is not None) & (punto3 is not None)):
                imagen_n = imagen[punto1[1]:punto3[1], punto1[0]:punto2[0]]
            elif(punto2 is not None):
                imagen_n = imagen[punto1[1]:alto_imagen-punto2[1], punto1[0]:punto2[0]]
            else:
                imagen_n = imagen[punto1[1]:punto3[1], punto1[0]:alto_imagen-punto1[0]]

        elif((punto1 is None) & (punto4 is not None)):
            #3 Casos posibles:
            #Tenemos (2,3,4), (2,4), (3,4)
            if((punto2 is not None) & (punto3 is not None)):
                imagen_n = imagen[punto2[1]:punto4[1], punto3[0]:punto4[0]]
            elif(punto2 is not None):
                imagen_n = imagen[punto2[1]:punto4[1], ancho_imagen-punto2[0]:punto4[0]]
            else:
                imagen_n = imagen[alto_imagen-punto3[1]:punto4[1], punto3[0]:punto4[0]]
        else:
            #Ni el punto 1 ni el 4, solo tengo 2 y 3
            imagen_n = imagen[punto2[1]:punto3[1], punto3[0]:punto2[0]]
            
   #Esta absurdez es por que necesito implementar la recomposición de alguno de los puntos, en caso de no ser detectados
    coordenadas_puntos = puntos
    return imagen_n, coordenadas_puntos



