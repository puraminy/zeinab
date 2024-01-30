from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch.nn as nn
from torchviz import make_dot

# Load data from CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Split and scale the data
# X = df[['flow_rate', 'conc_nano', 'Kfluid', 'heat_flux', 'X_D']]
X = df[['flow_rate', 'conc_nano', 'Kfluid', 'heat_flux', 'X_D']]

y = df['HTC'] # heat transfer Cofficent

# Split data to test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
# Fix scales
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Normalize inputs and targets to zero mean and unity standard deviation
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()
y_train = (y_train - y_train.mean()) / y_train.std()
y_test = (y_test - y_test.mean()) / y_test.std()

# Define the neural network
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Define the neural network with ReLU activation function
class ReLUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.linear = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.linear(x)
        return x

# Define the neural network with Tanh activation function
class TanhModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(10, 1)
        self.linear = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.linear(x)
        return x

# Define the neural network with Tanh activation function
class ReLUReLUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(10, 1)
        self.linear = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.linear(x)
        return x

models = [RegressionModel, ReLUModel, TanhModel, ReLUReLUModel]
# User input for selecting the model and number of epochs
model_type = input("Select the model " \
        + " ".join([str(i) + ")" + m.__name__ for i,m in enumerate(models)]) + ":")
num_epochs = input("Enter the number of epochs: [300]")
if not model_type:
    model_type = 0
else:
    model_type = int(model_type)

if not num_epochs: 
    num_epochs = 300 
else:
    num_epochs = int(num_epochs)

input_size = X_train.shape[1]
print("input size is:", input_size)
# Instantiate the selected model
if model_type > len(models):
    print("Invalid model selection. Please enter 1 to ", len(models) - 1)
    exit()

model_class = models[model_type]
# Instantiate the model and define loss function and optimizer
model = model_class(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    print("Predictions:")
    print(predictions)
    print("y_test:")
    print(y_test)
    mse = nn.MSELoss()(predictions, y_test)
    print(f'Mean Squared Error on Test Set: {mse.item()}')
    mae = nn.L1Loss()(predictions, y_test)
    print(f'Mean Absolute Error on Test Set: {mae.item()}')


from sklearn.metrics import r2_score

# Denormalize the predictions and y_test
predictions_denormalized = predictions * y_test.std() + y_test.mean()
y_test_denormalized = y_test * y_test.std() + y_test.mean()

# Convert predictions and y_test to NumPy arrays
predictions_np = predictions_denormalized.numpy()
y_test_np = y_test_denormalized.numpy()

# Calculate R-squared
r_squared = r2_score(y_test_np, predictions_np)
print(f'R-squared on Test Set: {r_squared}')



# Visualize the model

dummy_input = torch.randn(1, input_size)

dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
dot.render("mlp_structure", format="png", cleanup=True)


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_np, predictions_np, alpha=0.5)
plt.title('Predictions vs. Actual (Test Set)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(y_test_np, y_test_np, color='green', label='Actual', alpha=0.5)
plt.scatter(y_test_np, predictions_np, color='blue', label='Predicted', alpha=0.5)
plt.title('Actual vs. Predicted Values (Test Set)')
plt.xlabel('Values')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()