import sys
import os
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras.applications import VGG16
from keras.utils import to_categorical

def modelo():
    vgg=VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    cnn=Sequential()
    for capa in vgg.layers:
         cnn.add(capa)
    cnn.add(Flatten())
    cnn.add(Dense(13,activation='softmax'))
    
    return cnn
#Esto es para que no haya sesiones de Keras en background
K.clear_session()

#Donde están las imagenes de entrenamiento y validacion
data_entrenamiento = 'C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\data\\Entrenamiento'
data_validacion = 'C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\data\\Validacion'

#Parámetros:
#Numero de veces de interaccion sobre el set de datos
epocas=2
#Tamaño para procesar las imagenes
altura, longitud =224, 224
#Números de imagenes que le pasamos al pc a procesar en cada paso
batch_size = 32
#El numero de pasos por época
pasos = 163
pasos_validacion = 41
#Ajustar profundidades, la primera y la segunda
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3,3)
tamano_pool = (2,2)
#Las clases a clasificar
clases = 13
#Grado de aprendizaje
lr = 0.0005

##Preprocesamiento de imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
    #Esto es que le hacemos zoom, la giramos y de todo para entrenarlo
)

validacion_datagen = ImageDataGenerator(
    rescale=1./255
    #solo las rescalamos
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode = 'categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura,longitud),
    batch_size=batch_size,
    class_mode = 'categorical'
)

cnn = modelo()

#optimizar algoritmos
optimizer = keras.optimizers.Adam(learning_rate=lr)
cnn.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics =['accuracy'])
cnn.fit(
    imagen_entrenamiento,
    steps_per_epoch=pasos,
    epochs=epocas, 
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion
    )

dir = './modelo'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\modelo\\modelo.h5')
cnn.save_weights('C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\modelo\\pesos.h5')

