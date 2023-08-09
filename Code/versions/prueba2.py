import numpy as np
import cv2

def canny(image):
    """
    Canny edge detection with automatic thresholds.
    """
    sigma = 0.25
    v = np.median(image)
    img = cv2.medianBlur(image, 3)
    img = cv2.GaussianBlur(image, (3, 3), 0)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # return the edged image
    return cv2.Canny(img, lower, upper)

def lineas(canny_image, original_image, rho=1, theta=np.pi/360, threshold=30):
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

    return line_image

imagen = cv2.imread("C:\\Users\\sergi\\Desktop\\ProyectoChess\\Pictures\\foto6.jpg")
imagen = cv2.resize(imagen, (600, 600))

# Recortar cuadrados de 20x20 de cada esquina
esquina_superior_izquierda = imagen[:40, :40]
cv2.imshow("2",esquina_superior_izquierda)
cv2.waitKey(0)
esquina_superior_derecha = imagen[:40, -40:]
esquina_inferior_izquierda = imagen[-40:, :40]
esquina_inferior_derecha = imagen[-40:, -40:]

# Aplicar el algoritmo a cada cuadrado
canny_superior_izquierda = canny(esquina_superior_izquierda)
line_image_superior_izquierda = lineas(canny_superior_izquierda, esquina_superior_izquierda)

canny_superior_derecha = canny(esquina_superior_derecha)
line_image_superior_derecha = lineas(canny_superior_derecha, esquina_superior_derecha)

canny_inferior_izquierda = canny(esquina_inferior_izquierda)
line_image_inferior_izquierda = lineas(canny_inferior_izquierda, esquina_inferior_izquierda)

canny_inferior_derecha = canny(esquina_inferior_derecha)
line_image_inferior_derecha = lineas(canny_inferior_derecha, esquina_inferior_derecha)

# Mostrar los resultados
cv2.imshow("Esquina superior izquierda", line_image_superior_izquierda)
cv2.imshow("Esquina superior derecha", line_image_superior_derecha)
cv2.imshow("Esquina inferior izquierda", line_image_inferior_izquierda)
cv2.imshow("Esquina inferior derecha", line_image_inferior_derecha)
cv2.waitKey(0)
cv2.destroyAllWindows()
