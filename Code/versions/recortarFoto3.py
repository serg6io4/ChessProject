import cv2
import numpy as np


#cojo todas las líneas y empiezo a comprobar cuales de ellas crean intersecciones y cuales de ellas no(son paralelas)
def encontrar_intersecciones(lines):
    intersecciones = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            punto_interseccion = encontrar_punto_interseccion(x1, y1, x2, y2, x3, y3, x4, y4)
            if punto_interseccion is not None:
                intersecciones.append(punto_interseccion)
    return intersecciones

def encontrar_punto_interseccion(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calcular las coordenadas del punto de intersección utilizando la fórmula de intersección de dos líneas
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)#Obtengo las coordenadas de inicio y fin de ambas líneas
    if det != 0:#Me aseguro que las lineas no sean paralelas, para poder operar
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
        return int(x), int(y)
    else:#si son paralelas pues nada 
        return None
#Aritmética para calcular ángulos con la arcotangente
def calcular_angulo(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    dx = x2 - x1
    dy = y2 - y1
    return np.degrees(np.arctan2(dy, dx))

def detectar_lineas(imagen):
    # Convertir a escala de grises
    #gaussian = cv2.GaussianBlur(imagen, (3,3), 0)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Aplicar el detector de bordes Canny
    edges = cv2.Canny(gray, 70, 90, apertureSize=3)
    cv2.imshow("Bordes", edges)
    cv2.waitKey(0)
    # Aplicar la transformada de Hough para detectar líneas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=13, maxLineGap=5)

    # Encontrar las intersecciones de las líneas detectadas
    intersecciones = encontrar_intersecciones(lines)

    # Encontrar el punto de intersección con ángulo cercano a 90 grados(busco aquello que sea una esquina, por así decirlo)
    punto_90grados = None
    angulo_minimo = float('inf')
    for i in range(len(intersecciones)):
        for j in range(i + 1, len(intersecciones)):
            punto1 = intersecciones[i]
            punto2 = intersecciones[j]
            angulo = calcular_angulo(punto1, punto2)
            if abs(90 - angulo) < angulo_minimo:#umbral
                punto_90grados = punto1 if abs(90 - angulo) < angulo_minimo else punto_90grados
                angulo_minimo = abs(90 - angulo)

    #Por si quieres ver las líneas se descomenta
    # Dibujar las líneas detectadas en la imagen original
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Dibujar el punto de intersección con ángulo cercano a 90 grados
    if punto_90grados is not None:
        cv2.circle(imagen, punto_90grados, 5, (0, 255, 0), -1)

    # Mostrar la imagen el punto de intersección
    cv2.imshow("Líneas y Punto de Intersección", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return punto_90grados

def recortarFoto(img, square_tam):
    imagen = img
    cv2.imshow("transfromada", img)
    cv2.waitKey(0)
    cuadrado_tam = square_tam
    alto, ancho = imagen.shape[:2]
    puntos = []
    # Coordenadas del cuadrado a recortar
    coordenadas1 = [0, 0, cuadrado_tam, cuadrado_tam]
    coordenadas2 = [ancho - cuadrado_tam, 0, ancho, cuadrado_tam]
    coordenadas3 = [0, alto - cuadrado_tam, cuadrado_tam, alto]
    coordenadas4 = [ancho - cuadrado_tam, alto - cuadrado_tam, ancho, alto]

    # Llamar a la función para recortar y mostrar el cuadrado
    cuadrado1 = imagen[coordenadas1[1]:coordenadas1[3], coordenadas1[0]:coordenadas1[2]]
    cuadrado2 = imagen[coordenadas2[1]:coordenadas2[3], coordenadas2[0]:coordenadas2[2]]
    cuadrado3 = imagen[coordenadas3[1]:coordenadas3[3], coordenadas3[0]:coordenadas3[2]]
    cuadrado4 = imagen[coordenadas4[1]:coordenadas4[3], coordenadas4[0]:coordenadas4[2]]

    punto1 = detectar_lineas(cuadrado1)
    puntos.append(punto1)
    punto2 = detectar_lineas(cuadrado2)
    puntos.append(punto2)
    punto3 = detectar_lineas(cuadrado3)
    puntos.append(punto3)
    punto4 = detectar_lineas(cuadrado4)
    puntos.append(punto4)

    #Si algún punto es nulo, intento rescatar el recorte con la complementación de dos puntos
    puntos_validos = [punto for punto in puntos if punto is not None ]
    print(puntos_validos)
    recortada_original=None

    if len(puntos_validos)<=1:
        #Falta 3 o no hay puntos, no es recuperable
        print("No hay suficientes puntos")
    elif len(puntos_validos)>=2:
        #Faltan dos puntos, recuperable

        #Tengo el punto 1 y 4,  caso ideal
        if ((punto1 != None)& (punto4!=None)):
            coordenadas_originales1 = [coordenadas1[0] + punto1[0], coordenadas1[1] + punto1[1]]
            coordenadas_originales4 = [coordenadas4[0] + punto4[0], coordenadas4[1] + punto4[1]]
            recortada_original = imagen[coordenadas_originales1[1]:coordenadas_originales4[1],
                                        coordenadas_originales1[0]:coordenadas_originales4[0]]
        #Solo tengo el punto 1 y, el punto 2 o 3
        elif ((punto1!=None) & (punto4==None)):
            if(punto2==None):
                coordenadas_originales1 = [coordenadas1[0] + punto1[0], coordenadas1[1] + punto1[1]]
                coordenadas_originales3 = [coordenadas3[0] + punto3[0], coordenadas3[1] + punto3[1]]
                recortada_original = imagen[coordenadas_originales1[1]:coordenadas_originales3[1],
                                            coordenadas_originales1[0]:coordenadas_originales3[0]-(coordenadas_originales3[1]-coordenadas_originales1[1])]
            else:
                coordenadas_originales1 = [coordenadas1[0] + punto1[0], coordenadas1[1] + punto1[1]]
                coordenadas_originales2 = [coordenadas2[0] + punto2[0], coordenadas2[1] + punto2[1]]
                recortada_original = imagen[coordenadas_originales1[1]:(coordenadas_originales2[1]+coordenadas_originales2[0]-coordenadas_originales1[0]),
                                            coordenadas_originales1[0]:coordenadas_originales2[0]]
        #Solo tengo el punto 4 y, el punto 2 o 3
        elif((punto1==None) & (punto4!=None)):
            if(punto2==None):
                coordenadas_originales4 = [coordenadas4[0] + punto4[0], coordenadas4[1] + punto4[1]]
                coordenadas_originales3 = [coordenadas3[0] + punto3[0], coordenadas3[1] + punto3[1]]
                coordenadas_originales1 = [coordenadas3[0] + punto3[0],
                                           (coordenadas_originales3[1]-(coordenadas_originales4[1]-coordenadas_originales3[1]))]
                recortada_original = imagen[coordenadas_originales1[1]:coordenadas_originales4[1],coordenadas_originales1[0]:coordenadas_originales4[0]]
            else:
                coordenadas_originales4 = [coordenadas4[0] + punto4[0], coordenadas4[1] + punto4[1]]
                coordenadas_originales2 = [coordenadas2[0] + punto2[0], coordenadas2[1] + punto2[1]]
                coordenadas_originales1 = [coordenadas_originales2[0]-coordenadas_originales4[0]-coordenadas_originales2[0],
                                           coordenadas2[1] + punto2[1]]
                print(coordenadas_originales2)
                print(coordenadas_originales4)
                print(coordenadas_originales1)
                recortada_original = imagen[coordenadas_originales1[1]:coordenadas_originales4[1],
                                            coordenadas_originales1[0]:coordenadas_originales4[0]]
        #Solo tengo el punto 2 y 3
        elif((punto1==None) & (punto4==None)):
            coordenadas_originales2 = [coordenadas2[0] + punto2[0], coordenadas2[1] + punto2[1]]
            coordenadas_originales3 = [coordenadas3[0] + punto3[0], coordenadas3[1] + punto3[1]]
            recortada_original= imagen[coordenadas_originales2[1]: coordenadas_originales3[1], 
                                       coordenadas_originales3[0]:coordenadas_originales2[0]]
    else:
        #Esto es en el caso de que falte 1 punto o ninguno
        coordenadas_originales1 = [coordenadas1[0] + punto1[0], coordenadas1[1] + punto1[1]]
        coordenadas_originales2 = [coordenadas2[0] + punto2[0], coordenadas2[1] + punto2[1]]
        coordenadas_originales3 = [coordenadas3[0] + punto3[0], coordenadas3[1] + punto3[1]]
        coordenadas_originales4 = [coordenadas4[0] + punto4[0], coordenadas4[1] + punto4[1]]
        # Recortar la imagen original utilizando las coordenadas obtenidas
        recortada_original = imagen[coordenadas_originales1[1]:coordenadas_originales4[1],
                                    coordenadas_originales1[0]:coordenadas_originales4[0]]
    

    # Mostrar la imagen recortada original
    cv2.imshow("Imagen Recortada Original", recortada_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()