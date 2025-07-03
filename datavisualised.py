from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Convert images from [0, 255] range to [0.0, 1.0] and to PyTorch tensors
transform = transforms.ToTensor()

# Load the MNIST training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Wrap the datasets in DataLoaders to handle batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Retrieve the first batch from the training DataLoader
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

# Visualize the first image in the batch and display its label
plt.imshow(example_data[0][0], cmap='gray')
plt.title(f'Label: {example_targets[0]}')
plt.show()