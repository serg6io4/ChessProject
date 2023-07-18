import cv2

def recortar(imagen):
    """
    Obtención de las casilas establecidas por el ancho y alto de una imagen

    :param:  Imagen
    :return: Casillas recortadas de la imagen original
    """
    # Cargar la imagen del tablero de ajedrez
    imagen_tablero = imagen

    # Obtener las dimensiones de la imagen
    ancho, alto = imagen_tablero.shape[:2]

    # Calcular el tamaño de cada casilla (dividir entre 8, ya que el tablero tiene que ser casi perfecto linealmente)
    tamaño_casilla_x = ancho // 8
    tamaño_casilla_y = alto // 8

    tamaño_casilla_x = int(tamaño_casilla_x)
    tamaño_casilla_y = int(tamaño_casilla_y)

    contador = 1  # Para mantener un orden numérico en los nombres de archivo

    for fila in range(8):
        for columna in range(8):
            # Aquí obtengo las casillas según las coordenadas
            x1 = columna * tamaño_casilla_x
            y1 = fila * tamaño_casilla_y
            x2 = x1 + tamaño_casilla_x
            y2 = y1 + tamaño_casilla_y

            # Aquí recorto la casilla del punto actual
            casilla = imagen_tablero[y1:y2, x1:x2]
            
            # Los va a ir poniendo de uno en uno desde arriba hasta abajo
            nombre_archivo = f'{contador}.jpg'
            contador += 1

            # Guardo la casilla como una imagen separada
            cv2.imwrite(nombre_archivo, casilla)

#recortar(cv2.imread("C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\tablero.jpg"))