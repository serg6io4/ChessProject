import cv2
import numpy as np

def recortar_cuadrado(imagen, coordenadas):
    # Extraer las coordenadas
    x1, y1, x2, y2 = coordenadas

    # Recortar el cuadrado de la imagen
    cuadrado_recortado = imagen[y1:y2, x1:x2]

    return cuadrado_recortado

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
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det != 0:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
        return int(x), int(y)
    else:
        return None

def detectar_lineas(imagen):

    gaussian = cv2.GaussianBlur(imagen, (5,5), 0)
   
    # Convertir a escala de grises
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
   
    # Aplicar el detector de bordes Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow("edges",edges)
    cv2.waitKey(0)
    # Aplicar la transformada de Hough para detectar líneas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)

    # Encontrar las intersecciones de las líneas detectadas
    intersecciones = encontrar_intersecciones(lines)

    # Encontrar el punto de intersección más cercano al centro de la imagen
    centro_x, centro_y = imagen.shape[1] // 2, imagen.shape[0] // 2
    punto_cercano = None
    distancia_minima = float('inf')
    for punto in intersecciones:
        distancia = np.sqrt((punto[0] - centro_x) ** 2 + (punto[1] - centro_y) ** 2)
        if distancia < distancia_minima:
            punto_cercano = punto
            distancia_minima = distancia

    # Dibujar las líneas detectadas en la imagen original
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Dibujar el punto de intersección más cercano al centro de la imagen
    if punto_cercano is not None:
        cv2.circle(imagen, punto_cercano, 5, (0, 255, 0), -1)

    # Mostrar la imagen con la línea y el punto de intersección más cercano al centro
    cv2.imshow("Línea y Punto de Intersección", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return punto_cercano

def recortarFoto(img, square_tam):
    
    imagen = cv2.resize(img, (600,600))
    alto, ancho= imagen.shape[:2]
    cuadrado_tam=50
    # Coordenadas del cuadrado a recortar
    coordenadas1 = [0, 0, cuadrado_tam, cuadrado_tam]
    coordenadas2 = [ancho - cuadrado_tam, 0, ancho, cuadrado_tam]
    coordenadas3 = [0, alto - cuadrado_tam, cuadrado_tam, alto]
    coordenadas4 = [ancho - cuadrado_tam, alto - cuadrado_tam, ancho, alto]
    # Llamar a la función para recortar y mostrar el cuadrado
    cuadrado1 = recortar_cuadrado(imagen, coordenadas1)
    cuadrado2 = recortar_cuadrado(imagen, coordenadas2)
    cuadrado3 = recortar_cuadrado(imagen, coordenadas3)
    cuadrado4 = recortar_cuadrado(imagen, coordenadas4)
    punto1 = detectar_lineas(cuadrado1)
    punto2 = detectar_lineas(cuadrado2)
    punto3 = detectar_lineas(cuadrado3)
    punto4 = detectar_lineas(cuadrado4)

    # Obtener las coordenadas en la imagen original
    coordenadas_originales1 = [coordenadas1[0] + punto1[0], coordenadas1[1] + punto1[1]]
    coordenadas_originales2 = [coordenadas2[0] + punto2[0], coordenadas2[1] + punto2[1]]
    coordenadas_originales3 = [coordenadas3[0] + punto3[0], coordenadas3[1] + punto3[1]]
    coordenadas_originales4 = [coordenadas4[0] + punto4[0], coordenadas4[1] + punto4[1]]

    # Recortar la imagen original utilizando las coordenadas obtenidas
    recortada_original = imagen[min(coordenadas_originales1[1], coordenadas_originales3[1]):max(coordenadas_originales2[1], coordenadas_originales4[1]),
                                min(coordenadas_originales1[0], coordenadas_originales2[0]):max(coordenadas_originales3[0], coordenadas_originales4[0])]

    # Mostrar la imagen recortada original
    cv2.imshow("Imagen Recortada Original", recortada_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#recortarFoto('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\foto6.jpg', 20)