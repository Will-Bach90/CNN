# William Bach
# CSS 486 Autumn 2023

# spatial transformer code was derived and adapted from the following tutorial:
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

train_path = 'pneumonia-dataset/train'
test_path = 'pneumonia-dataset/test'
valid_path = 'pneumonia-dataset/val'

batch_size = 16
img_height = 500
img_width = 500
# Define transformations
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # Resize images
    transforms.Grayscale(),                      # Convert to grayscale if your images are grayscale
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))         # Normalize
])


# Train DataLoader
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation DataLoader
valid_dataset = datasets.ImageFolder(root=valid_path, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Test DataLoader
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SimpleCNNWithSTN(nn.Module):
    def __init__(self, img_height, img_width):
        super(SimpleCNNWithSTN, self).__init__()

        # Spatial Transformer Network (STN) Components

        # Localization Network: Consists of convolutional and pooling layers.
        # This network learns spatial transformations to be applied to the input image.
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),  # Convolutional layer with 8 filters
            nn.MaxPool2d(2, stride=2),                 # Max pooling layer to reduce spatial dimensions
            nn.ReLU(True),                             # ReLU activation function
            nn.Conv2d(8, 10, kernel_size=5, padding=2),# Another convolutional layer, this time with 10 filters
            nn.MaxPool2d(2, stride=2),                 # Another max pooling layer
            nn.ReLU(True)                              # ReLU activation function
        )

        # Regressor for the affine transformation matrix
        # This network outputs the parameters of the affine transformation.
        self.fc_loc = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the localization network
            nn.Linear(10 * img_height // 4 * img_width // 4, 32),  # Fully connected layer
            nn.ReLU(True),  # ReLU activation function
            nn.Linear(32, 3 * 2)  # Output layer for 6 parameters of the affine transformation
        )

        # Initialize the weights and bias with identity transformation
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Standard Convolutional Neural Network Components

        # Convolutional Layer: A single convolutional layer with 32 filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Max Pooling Layer: A pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layer: A dense layer for binary classification output
        self.fc1 = nn.Linear(32 * img_height // 2 * img_width // 2, 1)

    # Spatial Transformer Network (STN) forward function
    def stn(self, x):
        xs = self.localization(x)  # Apply the localization network
        xs = xs.view(-1, 10 * img_height // 4 * img_width // 4)  # Flatten the output
        theta = self.fc_loc(xs)  # Calculate the affine transformation parameters
        theta = theta.view(-1, 2, 3)  # Reshape theta to the transformation matrix

        grid = F.affine_grid(theta, x.size())  # Create the sampling grid
        x = F.grid_sample(x, grid)  # Sample the input image according to the grid
        return x

    def forward(self, x):
        # Transform the input using the STN
        x = self.stn(x)

        # Forward pass through the CNN
        x = F.relu(self.pool(self.conv1(x)))  # Apply convolution and pooling layers
        x = x.view(-1, 32 * img_height // 2 * img_width // 2)  # Flatten the output
        x = torch.sigmoid(self.fc1(x))  # Apply the fully connected layer and sigmoid activation
        return x



# Set image dimensions
img_height, img_width = 500, 500

# Instantiate the model and move it to the device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNNWithSTN(img_height, img_width).to(device)

# Stochastic Gradient Descent (SGD) optimizer is used with a learning rate of 0.01.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # Set the model to training mode

    # Loop over each batch from the training data
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move the data and target tensors to the specified device (GPU/CPU)

        optimizer.zero_grad()  # Clear previous gradients

        output = model(data)  # Forward pass: compute the model output for the current batch

        # Binary cross-entropy loss for binary classification tasks
        loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(1).float())

        loss.backward()  # Backward pass: compute the gradients of the loss

        optimizer.step()  # Update model parameters based on gradients

        # Print loss every 500 batches
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# Define the testing function
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode

    test_loss = 0  # Total test loss
    correct = 0  # Total correct predictions
    all_preds = []  # Store all predictions
    all_targets = []  # Store all targets

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up the batch loss using binary cross-entropy
            test_loss += F.binary_cross_entropy_with_logits(output, target.unsqueeze(1).float(), reduction='sum').item()

            # Calculate predictions and count correct ones
            pred = torch.round(torch.sigmoid(output))
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Extend the lists with predictions and targets for this batch
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

    # Calculate average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # Print test loss and accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    # Return test metrics and the lists of predictions and targets
    return test_loss, accuracy, np.array(all_preds), np.array(all_targets)


# Initialize parameters for early stopping
best_accuracy = 0  # Best accuracy observed so far
patience = 3       # Number of epochs to wait without improvement before stopping
no_improvement = 0 # Counter for epochs without improvement
model_save_path = 'stn_model.pt'  # Path where the best model will be saved

# Loop over epochs for training
for epoch in range(1, 26):  # Train for 25 epochs
    train(model, device, train_loader, optimizer, epoch)  # Train the model for one epoch
    test_loss, accuracy, predictions, targets = test(model, device, test_loader)  # Evaluate the model on the test set

    # Early stopping and model saving
    if accuracy > best_accuracy:
        best_accuracy = accuracy  # Update the best observed accuracy
        no_improvement = 0       # Reset the no_improvement counter
        torch.save(model.state_dict(), model_save_path)  # Save the model's state dictionary
        print(f"Saved model with accuracy: {accuracy:.4f}")
    else:
        no_improvement += 1  # Increment the no_improvement counter

    # Check if the patience limit is reached for early stopping
    if no_improvement >= patience:
        print("Early stopping triggered")
        break  # Exit the training loop

# After training, compute and print the confusion matrix
cm = confusion_matrix(targets, predictions)
# Class names 
class_names = ['Healthy', 'Pneumonia']

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, fmt='d', cmap='crest', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()