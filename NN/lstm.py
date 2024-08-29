import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load and normalize the training data
train_data = pd.read_csv('data/data_nn.csv')
scaler = MinMaxScaler()
train_data[['flowrate', 'temp', 'init conc', 'conc']] = scaler.fit_transform(train_data[['flowrate', 'temp', 'init conc', 'conc']])


def pad_sequences_to_longest(sequences, padding_value=0.0):
    max_time_steps = max(seq.shape[0] for seq in sequences)  # Maximum length across sequences
    max_features = max(seq.shape[1] for seq in sequences)    # Maximum number of features

    padded_sequences = []
    for seq in sequences:
        # Calculate the amount of padding needed for both dimensions
        pad_time = max_time_steps - seq.shape[0]
        pad_features = max_features - seq.shape[1]

        # Pad the sequence with padding_value
        padded_seq = np.pad(seq, ((0, pad_time), (0, pad_features)),
                            mode='constant', constant_values=padding_value)
        padded_sequences.append(padded_seq)

    return padded_sequences


def create_sequences(data):
    xs, ys = [], []
    start_idx = 0
    seq_length = 0
    idx = 0
    while start_idx + seq_length < len(data):
        if data.iloc[start_idx + seq_length, 0] < seq_length and idx > 0:
            start_idx += seq_length # Skip rest 
            seq_length = 0
            idx = start_idx
            continue

        if data.iloc[idx, -1] == 0:
            seq_length += 1
            idx += 1
            continue

        # Create the sequence
        x = data.iloc[start_idx:(start_idx + seq_length), :-1].values
        y = data.iloc[start_idx + seq_length, -1]
        xs.append(x)
        ys.append(y)

        start_idx += 1  # Move to the next time step
        idx += 1

    pxs = pad_sequences_to_longest(xs)
    X_tensor = torch.tensor(pxs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return  X_tensor, y_tensor

features = train_data[['time','flowrate', 'temp', 'init conc', 'conc']]
X, y = create_sequences(features)

# Load and prepare the test data
test_data = pd.read_csv('data/data_test_nn.csv')
test_data[['flowrate', 'temp', 'init conc', 'conc']] = scaler.transform(test_data[['flowrate', 'temp', 'init conc', 'conc']])

X_test_new, y_test_new = create_sequences(test_data[['time','flowrate', 'temp', 'init conc', 'conc']])

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.transform_func = nn.Identity()  

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.transform_func(out)  # Apply non-linear transformation function
        return out

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, optimizer
input_dim = 4  # time, flowrate, temp, init conc
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 1000
learning_rate = 0.01

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Move data to the device
X, y = X.to(device), y.to(device)
 
# Train the model
for epoch in range(num_epochs):
    model.train()
    output = model(X)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    # Ensure X_test_new is correctly shaped
    X_test_new = X_test_new.to(device)
    predictions = model(X_test_new).cpu().numpy()
    y_test_new_np = y_test_new.cpu().numpy()

    # Calculate metrics
    test_loss = mean_squared_error(y_test_new_np, predictions)
    r2 = r2_score(y_test_new_np, predictions)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'R² Score: {r2:.4f}')

    # Prepare DataFrame for output
    results = pd.DataFrame({
        'Actual': y_test_new_np.flatten(),
        'Predicted': predictions.flatten()
    })

    # Save the results to a CSV file
    results.to_csv('test_predictions.csv', index=False)

print('Test results saved to test_predictions.csv')

