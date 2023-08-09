import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

longitud, altura = 100, 100
modelo = 'C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\modelo\\modelo.h5'
pesos = 'C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\modelo\\pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    if respuesta == 0:
        print("BB")
    elif respuesta == 1:
        print("BW")
    elif respuesta == 2:
        print("Empty")
    elif respuesta == 3:
        print("KB")
    elif respuesta == 4:
        print("KW")
    elif respuesta == 5:
        print("KNB")
    elif respuesta == 6:
        print("KNW")
    elif respuesta == 7:
        print("PB")
    elif respuesta == 8:
        print("PW")
    elif respuesta == 9:
        print("QB")
    elif respuesta == 10:
        print("QW")
    elif respuesta == 11:
        print("RB")
    elif respuesta == 12:
        print("RW")
    

predict('C:\\Users\\sergi\Desktop\\ProyectoChess\\caballo negro.jpg')

