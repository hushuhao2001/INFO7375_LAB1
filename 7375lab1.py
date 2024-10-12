import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Task 1: Define the Multilayer Neural Network
class MultilayerNeuralNetwork(nn.Module):
    def __init__(self):
        super(MultilayerNeuralNetwork, self).__init__()

        # Define layers according to the specification
        self.layer1 = nn.Linear(20, 10)  # 20 input features -> 10 neurons (layer 1)
        self.layer2 = nn.Linear(10, 8)  # 10 neurons -> 8 neurons (layer 2)
        self.layer3 = nn.Linear(8, 8)  # 8 neurons -> 8 neurons (layer 3) (second layer of the (8/ReLU)²)
        self.layer4 = nn.Linear(8, 4)  # 8 neurons -> 4 neurons (layer 4)
        self.output = nn.Linear(4, 1)  # 4 neurons -> 1 neuron (output layer)

        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))  # Second layer of the (8/ReLU)²
        x = self.relu(self.layer4(x))
        x = self.sigmoid(self.output(x))
        return x


# Task 2: Develop a Training Set
# Create a binary classification dataset using sklearn
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the dataset (this is important for training neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the dataset into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
    1)  # Add an extra dimension for binary classification
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Instantiate the neural network model
model = MultilayerNeuralNetwork()

# Define the loss function (binary cross-entropy) and the optimizer (Adam)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
epochs = 100
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"\nTest Loss: {test_loss.item():.4f}")

    # Compute accuracy
    predicted = (test_outputs > 0.5).float()  # Apply threshold to get binary predictions
    accuracy = (predicted.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
