import os
import numpy as np
from shapely.geometry import Polygon

def poligono(ruta_txt):
    """
    Busca las coordenadas de un txt y transforma esas coordenadas a un poligono
    
    :param ruta_txt: ruta con coordenadas
    :return: poligono
    """
    with open(ruta_txt, 'r') as gt_f:
        gt_array = gt_f.readline().split()
        gt_array = np.array(gt_array).astype('int32')

        top_left_point = (gt_array[0], gt_array[1])
        bottom_left_point = (gt_array[2], gt_array[3])
        bottom_right_point = (gt_array[4], gt_array[5])
        top_right_point = (gt_array[6], gt_array[7])

        Poligono = Polygon([top_left_point, bottom_left_point, bottom_right_point, top_right_point])
        return Poligono

def IOU(Poligono1, Poligono2):
    """
    Compara el area de dos polígonos, realizando las uniones e intersecciones entre ambos.
    
    :param Poligono1: primer poligono
    :param Poligono2: segundo poligono
    :return: Porcentaje aproximado de similitud
    """
    intersect = Poligono1.intersection(Poligono2).area
    union = Poligono1.union(Poligono2).area
    iou = intersect / union
    return iou

# Parte de la fase de testeo
ruta_entrada = "Test\salida.txt"
ruta_salida = "Test\resultados_iou.txt"

with open(ruta_entrada, 'r') as file:
    imagen_txt_files = [line.strip() for line in file]

resultados_iou = []

for imagen in imagen_txt_files:
    ruta_original = os.path.join("transform_images/dataset", imagen + ".txt")
    ruta_prediccion = os.path.join("transform_images/dataset", imagen + "-colorgaussian-prediction.txt")
    
    if os.path.exists(ruta_prediccion):
        iou_result = IOU(poligono(ruta_original), poligono(ruta_prediccion))
        resultados_iou.append(str(iou_result))

with open(ruta_salida, 'w') as output_file:
    output_file.write("\n".join(resultados_iou) + "\n")







