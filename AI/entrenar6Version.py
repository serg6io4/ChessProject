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
    # Cargar el modelo MobileNetV2 preentrenado con la estrategia de inicialización 'spawn'
    torch.multiprocessing.set_start_method('spawn')
    model = models.mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 13)
    return model

if __name__ == "__main__":
    # Agregar la llamada a freeze_support() en el bloque if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    model = load_model()

    # Mover el modelo a la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Definir transformaciones para las imágenes (ajustar según tus necesidades)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Cargar los datasets
    data_root = 'C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\data'
    classes = os.listdir(data_root)

    all_image_paths = []
    labels = []

    for class_name in classes:
        class_path = os.path.join(data_root, class_name)
        image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        all_image_paths.extend(image_paths)
        labels.extend([classes.index(class_name)] * len(image_paths))

    train_image_paths, valid_image_paths, train_labels, valid_labels = train_test_split(
        all_image_paths, labels, test_size=0.2, random_state=42
    )

    train_dataset = CustomDataset(train_image_paths, train_labels, transform=transform)
    valid_dataset = CustomDataset(valid_image_paths, valid_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Entrenar el modelo
    num_epochs = 12

    # Listas para almacenar las métricas de entrenamiento y validación
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        model.train()
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

            # Calcular precisión en entrenamiento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

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

                # Calcular precisión en validación
                _, predicted_valid = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted_valid == labels).sum().item()

        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Accuracy: {valid_accuracy:.2f}%')

    # Supongamos que "model" es tu modelo entrenado
    ruta_concreta = "C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\modelo\\mobilenetv2_chess_classification.pt"
    os.makedirs(os.path.dirname(ruta_concreta), exist_ok=True)
    torch.save(model.state_dict(), ruta_concreta)

    # Calcular y visualizar la matriz de confusión
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Visualizar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Graficar la pérdida en entrenamiento y validación
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Graficar la precisión en entrenamiento y validación
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

