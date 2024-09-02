import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch.nn.utils.rnn as rnn_utils

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
    actual_lengths = []
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

    actual_lengths = [len(x) for x in xs]
    pxs = pad_sequences_to_longest(xs)
    X_tensor = torch.tensor(pxs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return  X_tensor, y_tensor, actual_lengths

features = train_data[['time','flowrate', 'temp', 'init conc', 'conc']]
X, y, train_lengths = create_sequences(features)

# Load and prepare the test data
test_data = pd.read_csv('data/data_test_nn.csv')
test_data[['flowrate', 'temp', 'init conc', 'conc']] = scaler.transform(test_data[['flowrate', 'temp', 'init conc', 'conc']])

X_test_new, y_test_new, test_lengths = create_sequences(test_data[['time','flowrate', 'temp', 'init conc', 'conc']])

model_seed = 123 # it is used for random_state of models
data_seed = 123 # it is used for random_state of splitting data into source and train sets
def set_model_seed(model_seed):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional
        self.transform_func = nn.LeakyReLU(negative_slope=0.01) 

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # Multiply by 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        packed_input = rnn_utils.pack_padded_sequence(x, lengths, 
                batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input, (h0, c0))
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        # Convert lengths to a tensor
        lengths = torch.tensor(lengths).to(x.device)

        # Use the output at the last time step for each sequence
        idx = torch.arange(0, output.size(0)).to(x.device)
        last_output = output[idx, lengths - 1]

        out = self.fc(last_output)
        out = self.transform_func(out)
        return out

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, optimizer

input_dim = 4
hidden_dim = 120
num_layers = 2  # Increased number of layers
output_dim = 1
num_epochs = 150
learning_rate = 0.0005  # Adjusted learning rate

# Train the model
# Function to train and evaluate the model multiple times
def train_and_evaluate_model(n_runs=5):
    r2_scores = []
    
    for run in range(n_runs):
        set_model_seed(model_seed + run)
        model = LSTM(input_dim, hidden_dim, num_layers, output_dim, dropout=0.2).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Move data to the device
        X_train, y_train = X.to(device), y.to(device)
 
        print("========================== Run:", run+1)
        for epoch in range(num_epochs):
            model.train()
            output = model(X_train, train_lengths)
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            # Ensure X_test_new is correctly shaped
            X_test = X_test_new.to(device)
            predictions = model(X_test, test_lengths).cpu().numpy()
            y_test_new_np = y_test_new.cpu().numpy()

            # Calculate metrics
            test_loss = mean_squared_error(y_test_new_np, predictions)
            r2 = r2_score(y_test_new_np, predictions)
            r2_scores.append(r2)

            print(f'Test Loss: {test_loss:.4f}')
            print(f'R² Score: {r2:.4f}')

            # Prepare DataFrame for output
            results = pd.DataFrame({
                'Actual': y_test_new_np.flatten(),
                'Predicted': predictions.flatten()
            })

            # Save the results to a CSV file
            results.to_csv(f'test_predictions_{run+1}.csv', index=False)
            print(results)
            print(f'Test results for run {run+1} saved to test_predictions_{run+1}.csv')
            print(f'Test Loss: {test_loss:.4f}')
            print(f'R² Score: {r2:.4f}')

    best_r2 = max(r2_scores)
    mean_r2 = np.mean(r2_scores)

    print(f'\nBest R² Score: {best_r2:.4f}')
    print(f'Mean R² Score over {n_runs} runs: {mean_r2:.4f}')

train_and_evaluate_model(n_runs=3)

