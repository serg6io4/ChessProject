import argparse
import os
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

def load_model():
    """
    carga un modelo MobileNetV2 preentrenado y modifica su capa final para que
    sea compatible con una tarea de clasificación con 13 clases distintas.

    :param:
    :return: modelo modificado se devuelve como resultado de la función.
    """
    #Para garantizar la creación segura de subprocesos y procesos
    torch.multiprocessing.set_start_method('spawn')
    #Se carga el modelo MobileNetV2 preentrenado desde la biblioteca
    model = models.mobilenet_v2(pretrained=True)
    # Esta línea obtiene el número de características de entrada de 
    # la penúltima capa del clasificador, que es la capa antes de la capa de salida
    in_features = model.classifier[1].in_features
    #Aquí se modifica la última capa del clasificador
    model.classifier[1] = torch.nn.Linear(in_features, 13)
    return model

def main(data_dir, save_dir):
    # Cargar el modelo
    model = load_model()

    # Utilizar la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Definir transformaciones de las imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    #Donde se almacenan las imagenes(dataset)
    data_root = data_dir
    classes = os.listdir(data_root)
    #Variables para las rutas de las imágenes y para las etiquetas de las clases
    all_image_paths = []
    labels = []

        #Cargar rutas de archivos de imágenes y sus etiquetas correspondientes, organizados por clases.
    for class_name in classes:
        class_path = os.path.join(data_root, class_name)
        image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        all_image_paths.extend(image_paths)
        labels.extend([classes.index(class_name)] * len(image_paths))

    #División de los datasets(entrenamiento, validación y test)
    train_image_paths, remaining_image_paths, train_labels, remaining_labels = train_test_split(
        all_image_paths, labels, test_size=0.2, random_state=42
    )
    valid_image_paths, test_image_paths, valid_labels, test_labels = train_test_split(
        remaining_image_paths, remaining_labels, test_size=0.5, random_state=42
    )

    train_dataset = CustomDataset(train_image_paths, train_labels, transform=transform)
    valid_dataset = CustomDataset(valid_image_paths, valid_labels, transform=transform)
    test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_epochs = 10
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        # model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
        #Cálculo de métricas de entrenamiento
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        #Modo de evaluación (validación)
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)

                _, predicted_valid = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted_valid == labels).sum().item()

        #Cálculo de métricas de validación
        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        #Impresión de resultados
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Accuracy: {valid_accuracy:.2f}%')

    #Para guardar el modelo
    ruta_concreta = save_dir + "\clasiffier.pt"
    os.makedirs(os.path.dirname(ruta_concreta), exist_ok=True)
    torch.save(model.state_dict(), ruta_concreta)

    #Para relizar la matrix de confusion con los datos del testeo
    model.eval()
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

    # Gráfico de matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Etiquetas Predichas')
    plt.ylabel('Etiquetas Reales')
    plt.title('Matriz de Confusión (Conjunto de Pruebas)')
    plt.show()

    # Gráficos de pérdida y precisión
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Pérdida en Entrenamiento')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Pérdida en Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Pérdida en Entrenamiento y Validación')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Precisión en Entrenamiento')
    plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Precisión en Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.title('Precisión en Entrenamiento y Validación')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate your model.")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("save_dir", help="Path to the directory for saving the model")
    args = parser.parse_args()

    main(args.data_dir, args.save_dir)
    main()
