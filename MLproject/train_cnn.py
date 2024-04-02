import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
import numpy as np
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
testset = datasets.USPS(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, config['conv1_out_channels'], kernel_size=config['conv1_kernel_size'], stride=1, padding=1)
        self.conv2 = nn.Conv2d(config['conv1_out_channels'], config['conv2_out_channels'], kernel_size=config['conv2_kernel_size'], stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(config['fc1_in_features'], config['fc1_out_features'])
        self.fc2 = nn.Linear(config['fc1_out_features'], 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10):
    total_steps = len(train_loader)
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            predictions = []
            true_labels = []
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.tolist())
                true_labels.extend(labels.tolist())

            accuracy = correct / total
            precision = precision_score(true_labels, predictions, average='macro')
            recall = recall_score(true_labels, predictions, average='macro')
            print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

configurations = [
    {'conv1_out_channels': 16, 'conv1_kernel_size': 3, 'conv2_out_channels': 32, 'conv2_kernel_size': 3, 'fc1_in_features': 32 * 4 * 4, 'fc1_out_features': 128},
    {'conv1_out_channels': 32, 'conv1_kernel_size': 5, 'conv2_out_channels': 64, 'conv2_kernel_size': 5, 'fc1_in_features': 64 * 4 * 4, 'fc1_out_features': 256},
    {'conv1_out_channels': 8, 'conv1_kernel_size': 3, 'conv2_out_channels': 16, 'conv2_kernel_size': 3, 'fc1_in_features': 16 * 4 * 4, 'fc1_out_features': 64}
]

cnn_model_1 = CNN(configurations[0])
cnn_model_2 = CNN(configurations[1])
cnn_model_3 = CNN(configurations[2])

criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model_1.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model_2.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model_3.parameters(), lr=0.001)

print("Training CNN...")
train_model(cnn_model_1, criterion, cnn_optimizer, trainloader, testloader)
train_model(cnn_model_2, criterion, cnn_optimizer, trainloader, testloader)
train_model(cnn_model_3, criterion, cnn_optimizer, trainloader, testloader)

