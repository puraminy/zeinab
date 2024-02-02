from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score
from tabulate import tabulate

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


# Function to apply model on data and generate predictions
# Return predictions, MSE and R-Squared
def apply_model(model_class, X_train, X_test, y_train, y_test, num_epochs):
   # Fix scales
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

   # Normalize inputs and targets to zero mean and unity standard deviation
    X_train_normalized = (X_train - X_train.mean()) / X_train.std()
    X_test_normalized = (X_test - X_test.mean()) / X_test.std()
    y_train_normalized = (y_train - y_train.mean()) / y_train.std()
    y_test_normalized = (y_test - y_test.mean()) / y_test.std()

    input_size = X_train.shape[1]
    print("input size is:", input_size)

    # Instantiate the model and define loss function and optimizer
    model = model_class(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_normalized)
        loss = criterion(outputs, y_train_normalized)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_normalized)
    
    mse = nn.MSELoss()(predictions, y_test_normalized)
    mae = nn.L1Loss()(predictions, y_test_normalized)
    # Denormalize the predictions and y_test
    predictions_denormalized = predictions * y_test.std() + y_test.mean()

    # Convert predictions and y_test to NumPy arrays
    predictions_np = predictions_denormalized.numpy()
    y_test_np = y_test.numpy()

    # Calculate R-squared
    r_squared = r2_score(y_test_np, predictions_np)
    # Return predictions, MSE and R-Squared
    return predictions_np, mse, r_squared

############################## Plot Resuts #######################
def plot_results(predictions_np, y_test, title, file_name):
    # Convert predictions and y_test to NumPy arrays
    y_test_np = y_test.to_numpy()
    # Calculate R-squared
    r_squared = r2_score(y_test_np, predictions_np)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, y_test_np, color='green', label='Actual', alpha=0.5)
    plt.scatter(y_test_np, predictions_np, color='blue', label='Predicted', alpha=0.5)
    plt.title(title + ' ' +  f' R-Squared:{r_squared:.2f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    # Remove comment if you don't want to show the plot
    plt.show()
    # Save the image of plot in images folder
    plt.savefig("images/"+ file_name, format="png")

############################ Feature Selection ####################
# Finds the combinaiton of features that best predict y
def FeatureSelection(model_class, data, inputs, output, num_epochs, seed=123):
    results = {}
    candidates = []
    last_max = -1
    y = data[output] 
    # Remove extra columns
    data = data.drop([output, 'HTC_ANN1', 'HTC_ANN2'], axis=1).copy()
    while(True):
        for x in data.drop(candidates, axis=1).columns:
            if len(candidates) == 0:
                features = [x]
            else:
                features = x + candidates 

            X = data[features]
            mean_r_squared, _, _, _ = repeat_apply(num_repeats=3, X, y, num_epochs)
            results[x] = mean_r_squared

        max_results =  max(results.values())
        max_results_key = max(results, key=results.get)

        if max_results > last_max:
            candidates.append(max_results_key)
            last_max = max_results

            print('step: ' + str(len(candidates)))
            print(candidates)
            print('Adjusted R2: ' + str(max_results))
            print('===============')
        else:
            break

    print('\n\n')
    print('elminated variables: ')
    print(set(df.drop(y, axis=1).columns).difference(candidates))

# Repeats an apply_model to get average of results
def repeat_apply(num_repeats, X, y, num_epochs):
    models_r_squared = {}
    models_mse = {}
    for i in range(num_repeats):
        # changing random_seed is important to get differnt train and test sets
        random_seed = i
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.2, random_state=random_seed) 
        for m_type in selected_models:
            # Instantiate the selected model
            model_class = models[m_type]
            model_name = model_names[m_type]
            predictions, mse,r_squared=apply_model(model_class, 
                    X_train, X_test, y_train, y_test, num_epochs)

            if not model_names in models_r_squared:
                models_r_squared[model_name] = []
                models_mse[model_name] = []

            models_r_squared[model_name].append(r_squared*100)
            models_mse[model_name].append(mse)

    return models_r_squared, models_mse

############################### Start of Program ###################
models = [RegressionModel, ReLUModel, TanhModel, ReLUReLUModel]
model_names=[model.__name__ for model in models]
# User input for selecting the model and number of epochs
model_type = input("\n".join([str(i) + ")" + name for i,name in enumerate(model_names)]) \
        + "\nSelect the model (enter to select all models) [all]:")

if not model_type:
    model_type = "all"

if model_type.lower() == "all":
    selected_models = range(len(models)) # chose all models
else:
    model_type = int(model_type)
    if model_type > len(models):
        print("Invalid model selection. Please enter 1 to ", len(models) - 1)
        exit()
    selected_models = [model_type]

print("Selected Models:", [model_names[i] for i in selected_models])
num_epochs = input("Enter the number of epochs: [300]")
if not num_epochs: 
    num_epochs = 300 
else:
    num_epochs = int(num_epochs)

num_repeats = input("Enter the number of repeating predictions: [3]")
if not num_repeats: 
    num_repeats = 3 
else:
    num_repeats = int(num_repeats)

# Load data from CSV file into a DataFrame
data = pd.read_csv('data.csv')

# Input and output 
inputs = ['flow_rate', 'conc_nano', 'Kfluid', 'heat_flux', 'X_D']
output = 'HTC'
# Read data into X and y
X = data[inputs]
y = data[output] 

# for all models apply model on data for 3 times and get predictions, mse and r_squared
models_r_squared, models_mse = repeat_apply(num_repeats, X, y, num_epochs)

models_mean_r_squared = {}
models_std_r_squared = {}
models_max_r_squared = {}
for model_name, values in models_r_squared.items():
    models_mean_r_squared[model_name] = np.mean(values) 
    models_std_r_squared[model_name] = np.std(values)
    index = np.argmax(values)
    maximum = values[index] # get maximum r_squared for a model
    models_max_r_squared[model_name] = (index, maximum)

for model_name, values in models_mse.items():
    models_mean_mse[model_name] = np.mean(values)

best_model_index = -1
best_r_squared = 0
best_mse = 0
results = []
best_predictions = None
for model_name in model_names:
    mean_r_squared = models_mean_r_squared[model_name]
    std_r_squared = models_std_r_squared[model_name]
    mean_mse = models_mean_mse[model_name]
    result = {
            "model":model_name, 
            "R2": round(mean_r_squared,1), 
            "MSE": round(mean_mse,2),
            "R2 std": round(std_r_squared, 1)
            }
    results.append(result)
    if mean_r_squared > best_r_squared:
        best_r_squared = mean_r_squared
        best_mse = mean_mse
        best_model = model_name
        best_predictions = predictions

# Creata a Table for results with two columns
results_table = pd.DataFrame(data=results)
latex_table=tabulate(results_table, headers='keys', 
        tablefmt='latex_raw', showindex=False)
print("=============== Latex Code for results =================")
table_env = f"""
    \\begin{{table*}}
        \centering
        {latex_table}
        \caption{{Resuls for different models}}
        \label{{table-results}}
    \end{{table*}}
    """
print(table_env)
print("=========================================================")
# Show results
print("============ Results for models =========================")
print(results_table)
print("=========================================================")
best_model = models[best_model_index]
best_model_name = model_names[best_model_index]
print("Best R-Squred", best_r_squared)
print("Best model with better R-Squred:", best_model_name) 

print("Predictions:")
print("Actual, Predicted")
for y,p in zip(y_test.to_numpy(), best_predictions):
   print(y, ",", p.round(2))

title = "Prediction of " + output + " using " + " ".join(inputs)
file_name = f"R-{best_r_squared:.2f}-" + best_model_name + "-" + output + "-using-".join(inputs) + ".png"
# Show the best results
plot_results(best_predictions, y_test, title, file_name)
print("Results were saved in results.csv and Images were saved in images folder")

results_df = pd.DataFrame(columns=["HTC", "Predictions"])
results_df["HTC"] = y_test
pred_list = [x[0] for x in best_predictions]
results_df["Predictions"] = pd.Series(pred_list)
results_df.to_csv("results.csv", index=False)


# Apply model to combination of inputs and select the best
FeatureSelection(best_model, data, inputs, output, num_epochs, seed)

# Visualize the model and save it on mlp_structure image
# dummy_input = torch.randn(1, input_size)
# from torchviz import make_dot
# dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
# dot.render("mlp_structure", format="png", cleanup=True)

