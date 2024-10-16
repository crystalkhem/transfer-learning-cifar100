import torch
import torch.nn as nn
import torchvision
import torchvision
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(.5,.5,.5)
])

train = torchvision.datasets.CIFAR10('./data', download=True, train=True, transform=trans)
test = torchvision.datasets.CIFAR10('./data', download=True, train=False, transform=trans)

train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

model = torchvision.models.resnet34(weights='ResNet34_Weights.DEFAULT')

optimizer = optim.Adam(model.parameters(), lr=.1)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for name, param in model.named_parameters():
#     if 'layer4' in name or 'fc' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# for param in model.parameters():
#     param.requires_grad = True

model_in = model.fc.in_features
model.fc = nn.Linear(model_in, 10)
model.to(device)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for epoch in range(epochs):
      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      scheduler.step()
      train_loss /= len(train_loader)
      print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}')

train(model, train_dataloader, optimizer, criterion, device)

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()  # Assuming a classification problem
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate average loss and accuracy
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    return test_loss, accuracy
evaluate_model(model, test_dataloader, device)


def epoch(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    train_loss = 0.0
    for epoch in range(epochs):
      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      scheduler.step()
      train_loss /= len(train_loader)
      print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}')


epoch(model, train_dataloader, optimizer, criterion, device, 5)

evaluate_model(model, test_dataloader, device)
