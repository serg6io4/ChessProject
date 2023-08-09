import cv2
import numpy as np

# Cargamos la imagen
imagen = cv2.imread('C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\Captura2.jpg')
#La pasamos a tonos grises para que Canny pueda usarlo
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicamos Canny
bordes = cv2.Canny(imagen, 150, 200, L2gradient=True)
cv2.imshow("bordes", bordes)
# Aplicamos la TRansformada de Hough para que detecte las líneas pero con algunos requisitos de las mismas
lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=280, minLineLength=100, maxLineGap=10)

# Seleccionar las líneas que formen un ángulo recto (similar a un cuadrado, o lo que se pueda, porque parece que no puede ser)
if lineas is not None:
    #Para esto usamos la distancia Euclídea, comparando la primera y sucesivas, con las siguientes lineas
    lineas_seleccionadas = []
    for i in range(len(lineas)):
        x1, y1, x2, y2 = lineas[i][0]
        longitud1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        for j in range(i+1, len(lineas)):
            x3, y3, x4, y4 = lineas[j][0]
            longitud2 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
            
            #Comparo si la longitud de las líneas y los ángulos entre ellas
            diferencia = np.abs(longitud1 - longitud2)
            umbral = 1  # Este es el umbral de la diferencia de longitudes entre líneas(es bajito porque si no pilla los bordes de fuera)
            
            # Se calcula el ángulo de las líneas
            angulo = np.abs(np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y4 - y3, x4 - x3))
            angulo_grados = np.degrees(angulo)
            
            # Luego se mira si el ángulo que forman está dentro de los 90º(y su umbral), si es que si las guardo
            if diferencia < umbral and np.abs(angulo_grados - 90) < 10:
                lineas_seleccionadas.append(lineas[i])
                lineas_seleccionadas.append(lineas[j])
    #Con esto se quedarán todas aquellas líneas que vayan formando cuadrados(o casi) y luego se mostrarán y se recotará

    # Esto es para dibujar las líneas que han pasado
    for linea in lineas_seleccionadas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    #Y ahora pillo las coordenadas de las líneas que necesito para determinar el tablero
    coordenadas_x = [x for linea in lineas_seleccionadas for x, _, _, _ in linea]
    coordenadas_y = [y for linea in lineas_seleccionadas for _, y, _, _ in linea]
    min_x = min(coordenadas_x)
    max_x = max(coordenadas_x)
    min_y = min(coordenadas_y)
    max_y = max(coordenadas_y)
    
    # Recorto la región que me interesa de la imagen original, según las coordenadas de esas líneas
    imagen_recortada = imagen[min_y:max_y, min_x:max_x]
    
    # Mostrar la imagen recortada resultante
    cv2.imshow('Imagen recortada', imagen_recortada)
    cv2.waitKey(0)#Para que no se me quite al momento de abrirla

