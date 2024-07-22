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
import os

list_epochs = [200, 100, 50 , 20]
best_epochs = 100 

exp_df = pd.DataFrame()

num_repeats = 5
# num_repeats shows the number of times to repeat the experiment to get its average values
model_seed = 123 # it is used for random_state of models
# changing model_seed generates different results for each run for each model
# The same model_seed produces the same results in each run which is good for reporducability of experiments
# Dont change it if you want to reproduce the same results
data_seed = 123 # it is used for random_state of splitting data into source and train sets
# changing it creates different source and train sets.
# Since the number of data is low changing it can largely affect the results
torch.manual_seed(model_seed)
np.random.seed(model_seed)
learning_rate = 0.05
# The learnign rate used in ANN
#hidden_size1 = 10
hidden_size1 = 15

hidden_size2 = 10

# the number of neurons in hidden layers

# https://alexlenail.me/NN-SVG/
# use the site above to draw the following network
#
hidden_sizes = [15, 10, 3 ]
# nn.ReLU(), nn.Tanh(), nn.Identity()
activations = [nn.ReLU(), nn.ReLU(), nn.Tanh()]  # Specify activations for hidden layers

list_hidden_sizes = [[15, 10, 3], [8, 4], [15, 5]]
normalization_type = "z_score"

##################### New Models
class RBFN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RBFN, self).__init__()
        hidden_size = hidden_sizes[0]
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(self.hidden)
        self.hidden_layers.append(self.output)

    def forward(self, x):
        # Compute the distances from input to centers
        distances = torch.cdist(x, self.hidden.weight, p=2)
        # Apply Gaussian function
        basis_functions = torch.exp(-distances**2)
        output = self.output(basis_functions)
        return output

class GRNN(nn.Module):
    def __init__(self, input_size, sigma=1.0):
        super(GRNN, self).__init__()
        self.pattern_layer = nn.Linear(input_size, input_size, bias=False)
        self.sigma = sigma

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(self.pattern_layer)

    def forward(self, x):
        # Compute the distances from input to pattern layer
        distances = torch.cdist(x, self.pattern_layer.weight.t())
        # Apply Gaussian function
        basis_functions = torch.exp(-distances**2 / (2 * self.sigma**2))
        # Summation layer
        summation_layer = basis_functions.sum(dim=1, keepdim=True)
        # Output layer
        output = (basis_functions @ self.pattern_layer.weight).sum(dim=1, keepdim=True) / summation_layer
        return output


class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(32 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

######################################################
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.activations = activations if activations is not None else []

        # Input to first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Intermediate hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Last hidden layer to output
        self.hidden_layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            act = self.activations[-1]
            if i < len(self.activations) and self.activations[i] is not None:
                act = self.activations[i]
            x = act(x)
        return x


class Linear1HiddenLayer(nn.Module):
    activation1 = None
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        #                          O1
        #   O1                     O2                  
        #   ...   5 x 10 edges    ...   10 x 1       O1 ---> y
        #   ...                   ...                 
        #   O5                    ...  
        #                         O10
        #
        #  input (5 features)     hidden1           output 
        #   5 neuron             10 neurons        1 neuron
        #
        self.input_to_hidden1 = nn.Linear(input_size, hidden_sizes[0])
        self.hidden1_to_output = nn.Linear(hidden_sizes[0], 1)

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(self.input_to_hidden1)
        self.hidden_layers.append(self.hidden1_to_output)

    def forward(self, x):
        # order of computation
        x = self.input_to_hidden1(x)
        if self.activation1 is not None:
            x = self.activation1(x)
        x = self.hidden1_to_output(x)
        return x

class Linear2HiddenLayer(nn.Module):
    activation1 = None
    activation2 = None
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        #     
        #   O1                     O1                O1
        #   ...    5 x 10 edges    ...   10 x 5      ...  5 x 1     O1 ---> y
        #   ...                    ...               O5
        #   O5                     O10    
        #      
        #
        #  input(5 features)     hidden1           hidden2        output 
        #                       10 neurons         5 neurons
        self.input_to_hidden1 = nn.Linear(input_size, hidden_sizes[0])
        self.hidden1_to_hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden2_to_output = nn.Linear(hidden_sizes[1], 1)

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(self.input_to_hidden1)
        self.hidden_layers.append(self.hidden1_to_hidden2)
        self.hidden_layers.append(self.hidden2_to_output)

    def forward(self, x):
        # order of computation
        x = self.input_to_hidden1(x)
        if self.activation1 is not None:
            x = self.activation1(x)
        x = self.hidden1_to_hidden2(x)
        if self.activation2 is not None:
            x = self.activation2(x)
        x = self.hidden2_to_output(x)
        return x

# NonLinear Model with 1 hidden layer with Tanjant Hyperbolic function as nonlinear function
# (Note it inherits from 2 hidden layer class above
class Tanh1HiddenLayer(Linear1HiddenLayer):
    activation1 = nn.Tanh()

# NonLinear Model with 2 hidden layers (Note it inherits from 2 hidden layer class above)
class Tanh2HiddenLayer(Linear2HiddenLayer):
    activation1 = nn.Tanh()
    activation2 = nn.Tanh()

# NonLinear Model with 1 hidden layer with Relu function as nonlinear function
# (Note it inherits from 2 hidden layer class above
class Relu1HiddenLayer(Linear1HiddenLayer):
    activation1 = nn.ReLU()

class Relu2HiddenLayer(Linear2HiddenLayer):
    activation1 = nn.ReLU()
    activation2 = nn.ReLU()


# Define the normalization function
def normalize(data, normalization_type):
    if normalization_type == 'z_score':
        return (data - data.mean()) / data.std()
    elif normalization_type == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    else:
        raise ValueError("Unsupported normalization type. Choose 'z_score' or 'minmax'.")

# Function to apply model on data and generate predictions
# Return predictions, MSE and R-Squared
import torch.nn.init as init

def fit_model(model_class, X_train, X_test, y_train, y_test, num_epochs, hidden_sizes, display_steps=False):
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

   # Normalize inputs and targets to zero mean and unity standard deviation
    X_train_normalized = normalize(X_train, normalization_type)
    X_test_normalized = normalize(X_test, normalization_type)
    y_train_normalized = normalize(y_train, normalization_type)
    y_test_normalized = normalize(y_test, normalization_type)

    input_size = X_train.shape[1]

    if model_class == GRNN:
        model = model_class(input_size)
    else:
        model = model_class(input_size, hidden_sizes)

    # Initialize weights
    def weights_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0)

    model.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_normalized)
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"NaN detected in outputs at epoch {epoch + 1}")
            return None, None, None, model

        loss = criterion(outputs, y_train_normalized)
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            print(f"NaN detected in loss at epoch {epoch + 1}")
            return None, None, None, model

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if (epoch + 1) % 10 == 0 and display_steps:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_normalized)
        
        # Check for NaN in predictions
        if torch.isnan(predictions).any():
            print(f"NaN detected in predictions")
            return None, None, None, model

    mse = nn.MSELoss()(predictions, y_test_normalized)
    mae = nn.L1Loss()(predictions, y_test_normalized)
    predictions_denormalized = predictions * y_test.std() + y_test.mean()

    predictions_np = predictions_denormalized.numpy().flatten()  # Ensure predictions are 1D array
    y_test_np = y_test.numpy().flatten()  # Ensure y_test is 1D array

    r2 = r2_score(y_test_np, predictions_np)
    return predictions_np, mse, r2, model

############################## Plot Resuts #######################
def plot_results(predictions_np, y_test, title, file_name, show_plot=False):
    # Convert predictions and y_test to NumPy arrays
    y_test_np = y_test.to_numpy()
    # Calculate R-squared
    r2 = r2_score(y_test_np, predictions_np)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_np, y_test_np, color='green', label='Actual', alpha=0.5)
    ax.scatter(y_test_np, predictions_np, color='blue', label='Predicted', alpha=0.5)
    ax.set_title(title + ' ' +  f' R-Squared:{r2:.2f}')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.grid(True)
    if show_plot is True:
        plt.show()
    # Save the image of plot in images folder
    fig.savefig(os.path.join("images", file_name), format="png")

################ Generate Latex Tables
def generate_latax_table(table, caption="table", label=""):
    def escape_latex_special_chars(text):
        text = str(text)
        return text.replace('_', '\\_')

    table = table.map(escape_latex_special_chars)

    latex_table=tabulate(table, headers='keys', 
            tablefmt='latex_raw', showindex=False)

    column_format = "p{12cm}r"  # Adjust width as needed
    # Replace the default column format with the desired one
    latex_table = latex_table.replace('tabular}{l', f'tabular}}{{{column_format}}}')

    # Add \hline correctly
    latex_table_lines = latex_table.splitlines()
    latex_table_lines.insert(1, '\\hline')
    latex_table_lines.append('\\hline')
    latex_table = '\n'.join(latex_table_lines)
   
    table_env = f"""
    \\begin{{table*}}
        \\centering
        {latex_table}
        \\caption{{{caption}}}
        \\label{{{label}}}
    \\end{{table*}}
    """

    return table_env

############################ Feature Selection ####################
# Selects the best combination of features by removing features one by one
# Search about Backward Feature Elimination 
# Sensitivity Analysis Functions

def weight_analysis(model_class, data, inputs, output, num_epochs):
    y = data[output]
    X_train, X_test, y_train, y_test = train_test_split(data[inputs], y, test_size=0.2, random_state=data_seed)
    input_size = X_train.shape[1]
    _,_,r2, model = fit_model(model_class, X_train, X_test, y_train, y_test, num_epochs, hidden_sizes, activations)
    print("R2:", r2)

    weight_importances = {}
    with torch.no_grad():
        for i, feature in enumerate(inputs):
            importance = np.abs(model.hidden_layers[0].weight[:, i].numpy()).mean()
            weight_importances[feature] = importance
    
    weight_table = pd.DataFrame(list(weight_importances.items()), columns=['Feature', 'Weight Importance'])
    return weight_table


def jackknife_sensitivity_analysis(model_class, data, inputs, output, num_epochs):
    y = data[output]
    base_model_r2, _, _, _, _, _ = repeat_fit_model(model_class, 
            num_repeats, data[inputs], y, num_epochs, hidden_sizes)
    sensitivities = {}
    variances = []
    _num_repeats = input("Number of repeat [10]:")
    _num_repeats = int(_num_repeats) if _num_repeats else 10
    print("Please wait ...")

    for input_feature in inputs:
        reduced_inputs = [f for f in inputs if f != input_feature]
        reduced_r2_list = []
        for _ in range(_num_repeats):
            reduced_r2, _, _, _, _, _ = repeat_fit_model(model_class,
                    1, data[reduced_inputs], y, num_epochs, hidden_sizes)
            reduced_r2_list.append(reduced_r2)
        reduced_r2_mean = np.mean(reduced_r2_list)
        sensitivity = base_model_r2 - reduced_r2_mean
        variance = np.var(reduced_r2_list)
        sensitivities[input_feature] = sensitivity
        variances.append(variance)

    sensitivity_table = pd.DataFrame(list(sensitivities.items()), columns=['Feature', 'Sensitivity'])
    sensitivity_table['Variance'] = variances
    return sensitivity_table

def backward_feature_elimination(model_class, data, inputs, output, num_epochs):
    y = data[output] 
    # Remove extra columns
    # data = data.drop([output, 'HTC_ANN1', 'HTC_ANN2'], axis=1).copy()
    data = data.drop([output, 'm'], axis=1).copy()
    X = data[inputs]
    # it uses num_repeats, num_epochs and model_seed as global variables
    # use _ for output of repeat_fit_model, which you don't need to use here
    mean_r2, _, _, _, _,_ = repeat_fit_model(model_class,
            num_repeats, X, y, num_epochs, hidden_sizes)
    best_r2 = mean_r2
    print("Using all features")
    print("Features:", inputs)
    print("R2:", mean_r2)
    candidates = inputs
    results = {}
    rows = [] # Rows of a table to show the features and the result
    rows.append({"features": ",".join(inputs), "R2": round(mean_r2,2)})
    while(True):
        for feature in candidates:
            # Selet other features except for current feature
            features = [f for f in candidates if f != feature]
            X = data[features]
            mean_r2, _, _, _,_,_ = repeat_fit_model(model_class, 
                    num_repeats, X, y, num_epochs, hidden_sizes)
            results[feature] = mean_r2
            print("---------------------------------------")
            # Results of removing the feature
            print("Removing:", feature)
            print("Features:", features)
            print("R2:", mean_r2)
            rows.append({"features": ",".join(features), "R2": round(mean_r2,2)})

        max_results =  max(results.values())
        max_results_key = max(results, key=results.get)

        if max_results > best_r2:
            candidates.remove(max_results_key)
            best_r2 = max_results
            print('=================================')
            print('step: ' + str(len(inputs) - len(candidates)))
            print(candidates)
            print('Best R2: ' + str(max_results))
            print('================================')
        else:
            break

    print('\n\n')
    print('=============== Final Features ==================')
    print('Selected Features: ')
    print(candidates)
    print('Final R2: ' + str(best_r2))
    print('Elminated Features: ')
    print(set(data.columns).difference(candidates))

    table = pd.DataFrame(data=rows)
    return table

# Finds the combinaiton of features that best predict y
# This method add features one by one
# Search about Forward Feature Selction
def forward_feature_selection(model_class, data, inputs, output, num_epochs):
    y = data[output] 
    # Remove extra columns
    data = data.drop([output, 'm'], axis=1).copy()
    candidates = []
    best_r2 = -1
    results = {}
    rows = [] # Rows of a table to show the features and the result
    while(True):
        # for all features except for the candidates
        for feature in data.drop(candidates, axis=1).columns:
            if len(candidates) == 0:
                features = [feature]
            else:
                features = [feature] + candidates 

            X = data[features]
            mean_r2, _, _, _,_,_ = repeat_fit_model(model_class, 
                    num_repeats, X, y, num_epochs, hidden_sizes)
            if mean_r2 is None:
                continue
            results[feature] = mean_r2
            print("--------------------------------")
            print("Adding feature ", feature)
            print("Features:", features)
            print("R2:", mean_r2)
            rows.append({"features": features, "R2": round(mean_r2,2)})

        max_results =  max(results.values())
        max_results_key = max(results, key=results.get)

        if max_results > best_r2:
            candidates.append(max_results_key)
            best_r2 = max_results
            print('=================================')
            print('step: ' + str(len(candidates)))
            print(candidates)
            print('Best R2: ' + str(max_results))
            print('================================')
        else:
            break

    print('\n\n')
    print('=============== Final Features ==================')
    print('Selected Features: ')
    print(candidates)
    print('Final R2: ' + str(best_r2))
    print('Elminated Features: ')
    print(set(data.columns).difference(candidates))

    table = pd.DataFrame(data=rows)
    return table


# Repeats an fit_model to get average of results
def repeat_fit_model(model_class, num_repeats, 
        X, y, num_epochs, hidden_sizes, display_steps=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.2, 
        random_state=data_seed) 
    r2_list = []
    mse_list = []
    max_r2 = 0
    best_preds = None
    for i in range(num_repeats):
        predictions, mse, r2, model = fit_model(model_class, 
                X_train, X_test, y_train, y_test, num_epochs, hidden_sizes, 
                display_steps=display_steps)
        if r2 is None:
            continue

        if r2 > max_r2:
            max_r2 = r2
            best_preds = predictions

        r2_list.append(r2*100)
        mse_list.append(mse)

    if display_steps:
        print(r2_list)

    mean_r2 = np.mean(r2_list) if r2_list else None
    mean_mse = np.mean(mse_list) if mse_list else None
    std_r2 = np.std(r2_list) if r2_list else None
    std_mse = np.std(mse_list) if mse_list else None

    return mean_r2, std_r2, mean_mse, best_preds, max_r2*100, r2_list

############################### Start of Program ###################
# Load data from data CSV file in data folder
data = pd.read_csv(os.path.join('data','data.csv'))

# Input features and output 
#inputs = ['flow_rate1', 'conc_nano1', 'Kfluid1', 'heat_flux1', 'X_D1','flow_rate2', 'conc_nano2', 'Kfluid2', 'heat_flux2', 'X_D2']
inputs = ['flow_rate1', 'conc_nano1', 'Kfluid1', 'heat_flux1', 'X_D1','flow_rate2', 'conc_nano2', 'Kfluid2', 'heat_flux2', 'X_D2']
output = 'HTC'
# Read data into X and y
X = data[inputs]
y = data[output] 

models = [
            Linear1HiddenLayer, 
            Linear2HiddenLayer,
            Tanh1HiddenLayer,
            Tanh2HiddenLayer,
            Relu1HiddenLayer,
            Relu2HiddenLayer,
            FFNN,
            RBFN,
            GRNN
         ]

model_names=[model.__name__ for model in models]
# User input for selecting the model and number of epochs
answer = input("\n".join([str(i) + ")" + name for i,name in enumerate(model_names)]) \
        + "\nSelect one or several models (separated with space) [all]:")

if not answer:
    answer = "all"

if answer.lower() == "all":
    selected_models = range(len(models)) # choose the index of all models
else:
    indexes = answer.split()
    selected_models = []
    for ind in indexes:
        model_index = int(ind)
        if model_index > len(models):
            print("Invalid model selection. Please enter all or 1 to ", len(models) - 1)
            exit()
        selected_models.append(model_index)

print("Selected Models:", [model_names[i] for i in selected_models])
best_model_index = selected_models[0]

answer = input(f"Enter the number of epochs [{list_epochs}] (0 to skip training):")
if answer != "0":
    if answer: 
       list_epochs = [int(a) for a in answer.split()]

    answer = input(f"Enter the hidden sizes [{list_hidden_sizes}]:")
    if answer: 
       list_hidden_sizes = []
       hs = answer.split("#")
       for ans in hs:
          ans = ans.strip()
          h = [int(a) for a in ans.split()]
          list_hidden_sizes.append(h)

    answer = input(f"Enter the number of repeating predictions [{num_repeats}]:")
    if answer: 
       num_repeats = int(answer)


    best_mean_r2 = -1000
    best_mse = -1000
    best_r2 = -1000
    best_epochs = -1
    results = []
    model_best_predictions = {}
    # for all models
    for model_index in selected_models:
        for num_epochs in list_epochs:
            for hidden_sizes in list_hidden_sizes:
                # Instantiate the selected model
                model_class = models[model_index]
                model_name = model_names[model_index]
                # Apply model on data for 3 times and get predictions, mse and r2
                mean_r2, std_r2, mean_mse, model_best_preds, max_r2, r2_list = repeat_fit_model(
                        model_class,
                        num_repeats, X, y, num_epochs, hidden_sizes, display_steps=True)

                # Keep best seed to generate the same predictions later
                model_best_predictions[model_name] = model_best_preds

                if max_r2 > best_r2:
                    best_r2 = max_r2

                if mean_r2 > best_mean_r2:
                    best_mean_r2 = mean_r2
                    best_mse = mean_mse
                    best_model_index = model_index
                    best_epochs = num_epochs
                
                total_nodes = sum(hidden_sizes)

                result = {
                        "model":model_name, 
                        "R2": round(mean_r2,1), 
                        "MSE": round(mean_mse,2),
                        "R2 std": round(std_r2, 1),
                        "R2 List:": [round(x, 1) for x in r2_list],
                        "hidden sizes": hidden_sizes,
                        "total hs": total_nodes,
                        "epochs": num_epochs,
                        }
                results.append(result)

    # Creata a Table for results
    results_table = pd.DataFrame(data=results)
    # Sort methods by R2
    results_table = results_table.sort_values(by = "R2", ascending=False)
    # Create and save latex code for table
    results_table_latex = generate_latax_table(results_table, 
            caption="Results of different models", label="models")
    with open(os.path.join("tables", "results.tex"), 'w') as f:
        print(results_table_latex, file=f)

    best_model = models[best_model_index]
    best_model_name = model_names[best_model_index]
    # Show results
    print("============ Results for models =========================")
    print(results_table)
    print("=========================================================")
    print("Best R-Squred:", best_r2)
    print("Best Mean R-Squred:", best_mean_r2)
    print("Best model with better mean R-Squred:", best_model_name) 

    results_table.to_csv("exp.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.2, 
        random_state=data_seed) 

    # Show and save the plot for best results
    best_predictions = model_best_predictions[best_model_name] 
    title = "Prediction of " + output + " using " + " ".join(inputs)+" with " + best_model_name 
    file_name = f"R2-{best_r2:.2f}-" + best_model_name + "-" + output + "-using-".join(inputs) + ".png"

    print("\n\n")
    print("Plot was saved in images folder")
    answer = input("Do you want to see them? [y]:") 
    if answer == "y" or answer == "yes":
        plot_results(best_predictions, y_test, title, file_name, show_plot=True)
    else:
        plot_results(best_predictions, y_test, title, file_name, show_plot=False)

    # Save results of predicitons in a file named results.csv
    results_df = pd.DataFrame(columns=["HTC", "predictions"])
    results_df["HTC"] = y_test
    results_df.rename(columns={"HTC": "actual"}, inplace=True)
    pred_list = [round(x,2) for x in best_predictions]
    results_df["predictions"] = pred_list # pd.Series(pred_list)
    results_df.to_csv("results.csv", index=False)
    print("Predictions of best model were saved in results.csv")
    answer = input("Do you want to see them? [y]:") 
    if answer == "y" or answer == "yes":
       print("======= Predictions of best model:", best_model_name)
       print(results_df)

best_model = models[best_model_index]
best_model_name = model_names[best_model_index]
while True:
    print("\n\n")
    print(f"================= Feature Selection ({best_model_name}:{best_epochs} epochs) ======")
    print("\nPlease select a feature selection or sensitivity analysis method:\n")
    print("1. Backward Feature Elimination")
    print("2. Forward Feature Selection")
    print("3. Weight Analysis")
    print("4. Jackknife Sensitivity Analysis (Node Deletion Sensitivity)")
    print("q. Quit")

    answer = input("Enter the number of the method you want to run (or 'q' to quit): ").strip().lower()

    if answer == '1':
        print("============================= Backward Feature Elimination =============")
        backward_table = backward_feature_elimination(best_model, data, 
                inputs, output, best_epochs)
        print("------------ backward feature elimination ---------------")
        print(backward_table)

        backward_table_latex = generate_latax_table(backward_table, caption="Results of Backward Feature Elimination", label="backward")
        with open(os.path.join("tables", "backward.tex"), 'w') as f:
            print(backward_table_latex, file=f)

    elif answer == '2':
        print("============================= Forward Feature Selection ================")
        forward_table = forward_feature_selection(best_model, data, inputs, output, best_epochs)
        print("\n")
        print("------------ forward feature selection ---------------")
        print(forward_table)
        forward_table_latex = generate_latax_table(forward_table, caption="Results of Forward Feature Selection for different features", label="forward")
        with open(os.path.join("tables", "forward.tex"), 'w') as f:
            print(forward_table_latex, file=f)

    elif answer == '3':
        print("============================= Weight Analysis =============")
        weight_table = weight_analysis(best_model, data, inputs, output, best_epochs)
        print("------------ weight analysis ---------------")
        weight_table = weight_table.sort_values(by='Weight Importance', 
                ascending=False)
        print("Most important features:")
        print(weight_table)
        weight_table_latex = generate_latax_table(weight_table, caption="Results of Weight Analysis", label="weight_analysis")
        with open(os.path.join("tables", "weight-analysis.tex"), 'w') as f:
            print(weight_table_latex, file=f)

    elif answer == '4':
        print("============================= Jackknife Sensitivity Analysis =============")
        jackknife_table = jackknife_sensitivity_analysis(best_model, 
                data, inputs, output, best_epochs)
        print("------------ jackknife sensitivity analysis ---------------")
        # Sort and display the most sensitive features
        jackknife_table = jackknife_table.sort_values(by='Sensitivity', ascending=False)
        print(jackknife_table)
        jackknife_table_latex = generate_latax_table(jackknife_table, caption="Results of Jackknife Sensitivity Analysis", label="jackknife")
        with open(os.path.join("tables", "jackknife.tex"), 'w') as f:
            print(jackknife_table_latex, file=f)

        # Focus on the top 5 most important features
        important_features = jackknife_table.head(5)['Feature'].tolist()
        print("\nTop 5 important features to focus on:")
        print(important_features)

        # Investigate features with negative sensitivity
        negative_sensitivity_features = jackknife_table[jackknife_table['Sensitivity'] < 0]
        print("\nFeatures with negative sensitivity (potentially redundant or harmful):")
        print(negative_sensitivity_features)

    elif answer == 'q':
        print("Exiting the feature selection and sensitivity analysis loop. Goodbye!")
        break
    else:
        print("Invalid choice, please try again.")
    print("\n----------------------------------------------------------")
    input("Press any key to return to main menu ...")



print("----------------------Important! READ -------------------")
print("images are saved in images folder")
print("latex code for tables are saved in tables folder")
print("predictions are saved in results.csv file")

# Visualize the model and save it on mlp_structure image
# dummy_input = torch.randn(1, input_size)
# from torchviz import make_dot
# dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
# dot.render("mlp_structure", format="png", cleanup=True)

