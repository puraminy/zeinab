import torch
import numpy as np
from models import LSTM

# Example data (replace with your data)
data = np.array([
    # m, time, flowrate, temp, init conc, conc
    [0, 1, 200, 30, 1000, 0],
    [0, 2, 200, 30, 1000, 0],
    [0, 3, 200, 30, 1000, 0],
    # ... (more data)
])

# Set parameters
sequence_length = 10  # Number of time steps in each sequence
input_size = 4        # Number of features (excluding 'conc')
batch_size = 32       # Number of sequences in each batch

# Prepare input sequences and target values
X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length, 2:-1])  # Exclude 'm' and 'conc' columns
    y.append(data[i+sequence_length-1, -1])    # Target is 'conc' at the end of the sequence

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Reshape X_tensor to match LSTM input: (batch_size, sequence_length, input_size)
X_tensor = X_tensor.view(-1, sequence_length, input_size)

# Feed data to the LSTM model
model = LSTM(input_size, [50, 1])  # Example model
output = model(X_tensor)

print(output)

