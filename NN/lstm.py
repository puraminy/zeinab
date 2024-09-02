import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from latex import *
from plot import *
import torch.nn.utils.rnn as rnn_utils
import os 
# Load and normalize the training data
train_data = pd.read_csv('data/data_nn.csv')
scaler = MinMaxScaler()
# Define input and output features explicitly
features = ['time','flowrate', 'temp', 'init conc', 'conc']
scale_features = ['flowrate', 'temp', 'init conc', 'conc']
input_features = ['time', 'flowrate', 'temp', 'init conc']
output_feature = 'conc'


train_data[scale_features] = scaler.fit_transform(train_data[scale_features])


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


def create_sequences(data, max_seq_len=0):
    xs, ys = [], []
    start_idx = 0
    seq_length = max_seq_len
    actual_lengths = []
    idx = 0
    while start_idx + seq_length < len(data):
        if data.iloc[start_idx + seq_length, 0] < seq_length and idx > 0:
            start_idx += seq_length # Skip rest 
            seq_length = max_seq_len
            idx = start_idx
            continue

        if max_seq_len <= 0:
            if idx < len(data) and data.iloc[idx][output_feature] == 0:
                seq_length += 1
                idx += 1
                continue

        x = data.iloc[start_idx:(start_idx + seq_length)][input_features].values
        y = data.iloc[start_idx + seq_length][output_feature]
        xs.append(x)
        ys.append(y)

        start_idx += 1  # Move to the next time step
        idx += 1

    actual_lengths = [len(x) for x in xs]
    pxs = pad_sequences_to_longest(xs)
    X_tensor = torch.tensor(pxs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return  X_tensor, y_tensor, actual_lengths


# Load and prepare the test data
test_data = pd.read_csv('data/data_test_nn.csv')
test_data[scale_features] = scaler.transform(test_data[scale_features])

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
num_layers = 3  # Increased number of layers
output_dim = 1
num_epochs = 100
learning_rate = 0.0005  # Adjusted learning rate
dropout= 0.2

# Train the model
# Function to train and evaluate the model multiple times
def train_and_evaluate_model(n_runs=5, 
        max_seq_len=0, 
        step_print=True,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_epochs=num_epochs):
    r2_scores = []
    max_r2 = 0
    X, y, train_lengths = create_sequences(train_data[features], max_seq_len)
    X_test_new, y_test_new, test_lengths = create_sequences(test_data[features], max_seq_len)
    
    for run in range(n_runs):
        set_model_seed(model_seed + run)
        model = LSTM(input_dim, hidden_dim, 
                num_layers, output_dim, dropout=dropout).to(device)
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
            if (epoch + 1) % 10 == 0 and step_print:
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
            if r2 > max_r2:
                max_r2 = r2
                results = pd.DataFrame({
                    'hidden_dim':hidden_dim,
                    'num_layers':num_layers,
                    'max_seq':max_seq_len,
                    'num_epochs':num_epochs,
                    'Actual': y_test_new_np.flatten(),
                    'Predicted': predictions.flatten()
                })

                # Save the results to a CSV file
                results.to_csv(os.path.join('lstm_preds',f'{max_seq_len}-pr-{r2:.4f}.csv'), 
                        index=False)
                title = f"R2-{r2:.4f}"
                file_name = os.path.join("lstm_plots", title + ".png")

                print("\n\n")
                print("Plot was saved in lstm_plots folder")
                plot_results(predictions, y_test_new.cpu(), title, 
                        file_name, show_plot=False)
                if step_print:
                    print(results)
                print(f'Test results for run {run+1} saved')
            print(f'Test Loss: {test_loss:.4f}')
            print(f'R² Score: {r2:.4f}')

    best_r2 = max(r2_scores)
    mean_r2 = np.mean(r2_scores)

    print(f'\nBest R² Score: {best_r2:.4f}')
    print(f'Mean R² Score over {n_runs} runs: {mean_r2:.4f}')
    return mean_r2

r2_list = []
var_list = []
for var in  range(15): #[""]:# [200,150,130,120]: # range(14): # [10,20,30]:
    print("-"*50, "var:", var)
    r2 = train_and_evaluate_model(n_runs=2, 
            max_seq_len=var, 
            step_print=False,
            num_epochs=num_epochs,
            num_layers=num_layers,
            dropout=dropout,
            hidden_dim=hidden_dim)
    r2_list.append(r2)
    var_list.append(var)

results = pd.DataFrame({
    'run': var_list, 
    'r2': r2_list 
})
results.to_csv(f'run.csv', index=False)
print(results)
print(f'Results saved to run.csv')

