import os
import argparse
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Grayscale

class CustomDataset(Dataset):
    """
    Esta clase permite cargar un conjunto de imágenes junto con sus etiquetas,
    y opcionalmente aplicar transformaciones a las imágenes

    :param: Una lista de rutas de archivos de imagen, Una lista de etiquetas correspondientes a las imágenes, 
            Una transformación opcional que se puede aplicar a las imágenes
    :return: Retorna la imagen transformada (o sin transformar si no hay transformaciones) y la etiqueta correspondiente.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def cargar_modelo(ruta_modelo):
    # Cargar el modelo previamente entrenado
    model = models.mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 13)

    # Cargar los pesos del modelo entrenado
    model.load_state_dict(torch.load(ruta_modelo, map_location=torch.device('cpu')))
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using the test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = cargar_modelo(args.model_path)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    classes = os.listdir(args.data_root)

    all_image_paths = []
    labels = []

    for class_name in classes:
        class_path = os.path.join(args.data_root, class_name)
        image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        all_image_paths.extend(image_paths)
        labels.extend([classes.index(class_name)] * len(image_paths))

    test_image_paths, _, test_labels, _ = train_test_split(
        all_image_paths, labels, test_size=0.5, random_state=42
    )

    test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()

