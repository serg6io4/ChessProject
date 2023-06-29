import cv2
def recortar(imagen):
    # Cargar la imagen del tablero de ajedrez
    imagen_tablero = imagen

    # Obtener las dimensiones de la imagen
    ancho, alto, _ = imagen_tablero.shape

    # Calcular el tamaño de cada casilla(dividir entre 8, ya que el tablero tiene que ser casi perfecto linealmente)
    tamaño_casilla_x = ancho // 8
    tamaño_casilla_y = alto // 8

    tamaño_casilla_x = int(tamaño_casilla_x)
    tamaño_casilla_y = int(tamaño_casilla_y)

    for fila in range(8):
        for columna in range(8):
            # Aquí obtengo las casillas de según las coordenadas
            x1 = columna * tamaño_casilla_x
            y1 = (8 - 1 - fila) * tamaño_casilla_y  # Invierto el oden de las filas para que vaya de A8 hasta H8 y sucesivo
            x2 = x1 + tamaño_casilla_x
            y2 = y1 + tamaño_casilla_y

            # Aquí recorto la casilla del punto actual
            casilla = imagen_tablero[y1:y2, x1:x2]
            
            # Calculo el nombre de la casilla en el orden correcto(por ejemplo, A8, B8,C8, ... y sucesivos)
            letra_columna = chr(65 + columna)
            numero_fila = str(8 - fila)
            nombre_archivo = f'{letra_columna}{numero_fila}.jpg'#para darle un nombre antes de se guardada por cv2
            
            # Guardo la casilla como una imagen separada
            cv2.imwrite(nombre_archivo, casilla)
