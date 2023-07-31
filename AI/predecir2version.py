import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Cargar el modelo previamente entrenado
model = models.mobilenet_v2(pretrained=False)  # Cambiar a "False" para que no cargue los pesos preentrenados
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 13)

# Cargar los pesos del modelo entrenado
ruta_concreta = "C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\modelo\\mobilenetv2_chess_classification.pt"
model.load_state_dict(torch.load(ruta_concreta))
model.eval()

# Definir transformación para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Tomar la foto de entrada (asegúrate de que la ruta de la foto sea correcta)
ruta_foto = "C:\\Users\\sergi\\Desktop\\ProyectoChess\\rey_negro.jpg"
imagen = Image.open(ruta_foto)

# Preprocesar la imagen para que coincida con las transformaciones utilizadas durante el entrenamiento
imagen_preprocesada = transform(imagen)
imagen_preprocesada = imagen_preprocesada.unsqueeze(0)  # Agregar una dimensión extra para el tamaño del batch

# Pasar la imagen preprocesada a través del modelo para obtener las predicciones
with torch.no_grad():
    outputs = model(imagen_preprocesada)

# Interpretar las predicciones para obtener la clase predicha
_, predicted_class = torch.max(outputs, 1)

# Suponiendo que tienes un diccionario que mapea el índice de clase a su nombre
# (por ejemplo, si la clase 0 es "peon", la clase 1 es "torre", etc.)
clases = {0: "BB", 1: "BW", 2: "Empty", 3: "KB", 4: "KW", 5: "KNB", 6: "KNW", 7: "PB", 8: "PW", 9: "QB", 10: "QW", 11: "RB", 12: "RW"}

clase_predicha = clases[predicted_class.item()]
print("La foto de entrada ha sido clasificada como:", clase_predicha)
