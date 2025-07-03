import torch
import torch.nn as nn
import torch.nn.functional as F

#Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Fully connected layers:
        #Input 28*28 image (flattened to 784)
        #Hiddem layers: 128 and 64 neurons
        #Output: 10 classes (digits 0-9)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    #Define how data flows through the network
    def forward(self, x):
        #Flatten the image into a 1D vector
        x = x.view(-1, 28 * 28)

        #Apply linear layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #Final output (raw logits for 10 classes)
        x = self.fc3(x)
        return x
