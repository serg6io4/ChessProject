import os
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

def load_model():
    # Cargar el modelo MobileNetV2 preentrenado con la estrategia de inicialización 'spawn'
    torch.multiprocessing.set_start_method('spawn')
    model = models.mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 13)
    return model

if __name__ == "__main__":
    # Agregar la llamada a freeze_support() en el bloque if __name__ == '__main__':
    mp.freeze_support()

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
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Cargar los datasets
    train_dataset = datasets.ImageFolder('C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\data\\train', transform=transform)
    valid_dataset = datasets.ImageFolder('C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\data\\val', transform=transform)

    # Crear los Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    #Entrenar el modelo
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Calcular la pérdida de entrenamiento promedio para la época
        train_loss /= len(train_loader.dataset)

        # Validación
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calcular la pérdida de validación promedio y la precisión
        valid_loss /= len(valid_loader.dataset)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Supongamos que "model" es tu modelo entrenado
    ruta_concreta = "C:\\Users\\sergi\\Desktop\\ProyectoChess\\AI\\modelo\\mobilenetv2_chess_classification.pt"
    os.makedirs(os.path.dirname(ruta_concreta), exist_ok=True)
    torch.save(model.state_dict(), ruta_concreta)





