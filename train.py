import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import Net
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Transform images to tensors and normalize from [0, 255] to [0.0, 1.0]
transform = transforms.ToTensor()

#Load the training dataset and wrap it in a DataLoader with batching and shuffling
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#Load the test dataset and wrap it in a DataLoader (no shuffling)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Initialize the neural network, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Set number of epochs and start training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0

    #Training loop over batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad() #Reset gradients from previous step
        outputs = model(images) #Forward pass: model makes predictions
        loss = criterion(outputs, labels) #Compute loss between predictions and true labels
        loss.backward() #Backward pass: compute gradients
        optimizer.step() #Update model weights and biases
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} - Loss: {avg_loss:.4f}') #Print train loss

    #Evaluate model on test data
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): #Disable gradient computation for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct/ total
    print(f'Epoch {epoch + 1} - Accuracy:{accuracy:.2f}%') #Print test accuracy
    model.train() #Return to training mode

#Generate predictions for the entire test set after training
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

#Create and visualize the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
