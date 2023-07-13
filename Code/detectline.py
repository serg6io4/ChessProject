import numpy as np
import cv2

def canny(image):
    #Cambio temático e implemento un detector de Canny con umbrales automáticos
    sigma = 0.25
    v = np.median(image)
    img = cv2.medianBlur(image, 3)
    img = cv2.GaussianBlur(image, (3, 3), 0)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(img, lower, upper)

def lineas(canny_image, original_image, rho=1, theta=np.pi/360, threshold=200):
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
        cv2.circle(line_image, punto, 5, (255, 0, 0), -1)

    cv2.imshow("Líneas detectadas", line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return line_image, puntos_interseccion

def punto_esquina(x, y, puntos_interseccion, esquina_x, esquina_y):

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


def recortar_pre(imagen, coordenadas):
    Canny = canny(imagen)
    cv2.imshow("C", Canny)
    cv2.waitKey(0)
    alto_imagen, ancho_imagen = imagen.shape[:2]
    #No cogemos lineas, por el simple hecho que las lineas el medio para obtener los puntos
    lines, puntos = lineas(Canny,imagen)
    punto1 = punto_esquina(0, 0, puntos, 40, 40)
    punto2 = punto_esquina(ancho_imagen - 40, 0, puntos, 40, 40)
    punto3 = punto_esquina(0, alto_imagen - 40, puntos, 40, 40)
    punto4 = punto_esquina(ancho_imagen - 40, alto_imagen - 40, puntos, 40, 40)

    #Debido a que no siempre va a estar el punto 1 y el 4, tenemos que realizar un sistema el cual pueda recrear los puntos faltantes
    #Para ello será necesario, realizar operaciones con los sistemas de coordenadas con los puntos que tenemos
    puntos = []
    puntos.extend([punto1, punto2, punto3, punto4])
    #Esto lo realizo para saber cuantos puntos me faltan
    puntos_validos = [punto for punto in puntos if punto is not None ]
    imagen_n = None

    if len(puntos_validos<=1):
        print("No hay suficientes puntos para detectar el tablero")
    elif len(puntos_validos>2):
        if(punto1!=None & punto4!=None):
            #Realmente solo importa a la hora de recortar estas dos coordenadas
            imagen_n = imagen[punto1[1]:punto4[1], punto1[0]:punto4[0]]
        elif(punto1!=None & punto4==None):
            #3 Casos posibles:
            #Tenemos (1,2), (1,3), (1,2,3)
            print(" A Completar")
        elif(punto1==None & punto4 != None):
            #3 Casos posibles:
            #Tenemos (2,4), (3,4), (2,3,4)
            print(" A Completar")
        

    cv2.imshow("Previsualizacion", imagen_n)
    cv2.waitKey(0)
    return imagen



