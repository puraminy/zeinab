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

num_epochs = 500 
#num_epochs = 300 

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

hidden_size3 = 5
# the number of neurons in hidden layers

# https://alexlenail.me/NN-SVG/
# use the site above to draw the following network
#
class Linear1HiddenLayer(nn.Module):
    activation1 = None
    def __init__(self, input_size):
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
        self.input_to_hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden1_to_output = nn.Linear(hidden_size1, 1)

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
    def __init__(self, input_size):
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
        self.input_to_hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden1_to_hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.hidden2_to_output = nn.Linear(hidden_size2, 1)

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

# Function to apply model on data and generate predictions
# Return predictions, MSE and R-Squared
def fit_model(model_class, X_train, X_test, y_train, y_test, num_epochs, 
        display_steps=False):
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
    # Instantiate the model and define loss function and optimizer
    model = model_class(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_normalized)
        loss = criterion(outputs, y_train_normalized)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 and display_steps:
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
    r2 = r2_score(y_test_np, predictions_np)
    # Return predictions, MSE and R-Squared
    return predictions_np, mse, r2

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
    latex_table=tabulate(table, headers='keys', 
            tablefmt='latex_raw', showindex=False)
    table_env = f"""
        \\begin{{table*}}
            \centering
            {latex_table}
            \caption{{{caption}}}
            \label{{table-{label}}}
        \end{{table*}}
        """
    return table_env

############################ Feature Selection ####################
# Selects the best combination of features by removing features one by one
# Search about Backward Feature Elimination 
def backward_feature_elimination(model_class, data, inputs, output):
    y = data[output] 
    # Remove extra columns
    data = data.drop([output, 'HTC_ANN1', 'HTC_ANN2'], axis=1).copy()
    X = data[inputs]
    # it uses num_repeats, num_epochs and model_seed as global variables
    # use _ for output of repeat_fit_model, which you don't need to use here
    mean_r2, _, _, _, _,_ = repeat_fit_model(num_repeats, X, y, num_epochs)
    best_r2 = mean_r2
    print("Using all features")
    print("Features:", inputs)
    print("R2:", mean_r2)
    candidates = inputs
    results = {}
    rows = [] # Rows of a table to show the features and the result
    rows.append({"features": inputs, "R2": round(mean_r2,2)})
    while(True):
        for feature in candidates:
            # Selet other features except for current feature
            features = [f for f in candidates if f != feature]
            X = data[features]
            mean_r2, _, _, _,_,_ = repeat_fit_model(num_repeats, X, y, num_epochs)
            results[feature] = mean_r2
            print("---------------------------------------")
            # Results of removing the feature
            print("Removing:", feature)
            print("Features:", features)
            print("R2:", mean_r2)
            rows.append({"features": features, "R2": round(mean_r2,2)})

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

def forward_feature_selection(model_class, data, inputs, output):
    y = data[output] 
    # Remove extra columns
    data = data.drop([output, 'HTC_ANN1', 'HTC_ANN2'], axis=1).copy()
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
            mean_r2, _, _, _,_,_ = repeat_fit_model(num_repeats, X, y, num_epochs)
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
def repeat_fit_model(num_repeats, X, y, num_epochs, display_steps=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.2, 
        random_state=data_seed) 
    r2_list = []
    mse_list = []
    max_r2 = 0
    best_preds = None
    for i in range(num_repeats):
        predictions, mse, r2 = fit_model(model_class, 
                X_train, X_test, y_train, y_test, num_epochs, display_steps)

        if r2 > max_r2:
            max_r2 = r2
            best_preds = predictions

        r2_list.append(r2*100)
        mse_list.append(mse)

    if display_steps:
        print(r2_list)

    mean_r2 = np.mean(r2_list) 
    mean_mse = np.mean(mse_list)
    std_r2 = np.std(r2_list)
    std_mse = np.std(mse_list)

    return mean_r2, std_r2, mean_mse, best_preds, max_r2*100, r2_list

############################### Start of Program ###################

models = [
            Linear1HiddenLayer, 
            Linear2HiddenLayer,
            Tanh1HiddenLayer,
            Tanh2HiddenLayer,
            Relu1HiddenLayer,
            Relu2HiddenLayer
         ]
model_names=[model.__name__ for model in models]
# User input for selecting the model and number of epochs
answer = input("\n".join([str(i) + ")" + name for i,name in enumerate(model_names)]) \
        + "\nSelect the model (enter to select all models) [all]:")

if not answer:
    answer = "all"

if answer.lower() == "all":
    selected_models = range(len(models)) # choose the index of all models
else:
    model_index = int(answer)
    if model_index > len(models):
        print("Invalid model selection. Please enter all or 1 to ", len(models) - 1)
        exit()
    selected_models = [model_index]

print("Selected Models:", [model_names[i] for i in selected_models])
answer = input(f"Enter the number of epochs [{num_epochs}]:")
if answer: 
   num_epochs = int(answer)

answer = input(f"Enter the number of repeating predictions [{num_repeats}]:")
if answer: 
   num_repeats = int(answer)

# Load data from data CSV file in data folder
data = pd.read_csv(os.path.join('data','data.csv'))

# Input features and output 
inputs = ['flow_rate1', 'conc_nano1', 'Kfluid1', 'heat_flux1', 'X_D1','flow_rate2', 'conc_nano2', 'Kfluid2', 'heat_flux2', 'X_D2']
output = 'HTC'
# Read data into X and y
X = data[inputs]
y = data[output] 

best_model_index = -1
best_mean_r2 = 0
best_mse = 0
best_r2 = 0
results = []
model_best_predictions = {}
# for all models
for model_index in selected_models:
    # Instantiate the selected model
    model_class = models[model_index]
    model_name = model_names[model_index]
    # Apply model on data for 3 times and get predictions, mse and r2
    mean_r2, std_r2, mean_mse, model_best_preds, max_r2, r2_list = repeat_fit_model(
            num_repeats, X, y, num_epochs, display_steps=True)

    # Keep best seed to generate the same predictions later
    model_best_predictions[model_name] = model_best_preds

    if max_r2 > best_r2:
        best_r2 = max_r2

    if mean_r2 > best_mean_r2:
        best_mean_r2 = mean_r2
        best_mse = mean_mse
        best_model_index = model_index

    result = {
            "model":model_name, 
            "R2": round(mean_r2,1), 
            "MSE": round(mean_mse,2),
            "R2 std": round(std_r2, 1),
            "R2 List:": [round(x, 1) for x in r2_list]
            }
    results.append(result)

# Creata a Table for results
results_table = pd.DataFrame(data=results)
# Create and save latex code for table
results_table_latex = generate_latax_table(results_table, caption="Results of different models", label="models")
with open(os.path.join("tables", "results_table_latex.txt"), 'w') as f:
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
results_df = pd.DataFrame(columns=["HTC", "Predictions"])
results_df["HTC"] = y_test
pred_list = [round(x[0]) for x in best_predictions]
results_df["Predictions"] = pred_list # pd.Series(pred_list)
results_df.to_csv("results.csv", index=False)
print("Predictions of best model were saved in results.csv")
answer = input("Do you want to see them? [y]:") 
if answer == "y" or answer == "yes":
   print("======= Predictions of best model:", best_model_name)
   print(results_df)
print("\n\n")
answer = input("Do you want to run feature selections? [y/n]:")
if answer.lower() != "y" and answer.lower() != "yes":
    print("You selected no")
    exit()

# Apply model to combination of inputs and select the best
print("============================= Backward Feature Elimination =============")
backward_table = backward_feature_elimination(best_model, data, inputs, output)
print("============================= Forward Feature Selection ================")
forward_table = forward_feature_selection(best_model, data, inputs, output)

print("------------ backward freature elimination ---------------")
print(backward_table)

backward_table_latex = generate_latax_table(backward_table, caption="Results of Backward Feature Elimination", label="backward")
with open(os.path.join("tables", "backward_table_latex.txt"), 'w') as f:
    print(backward_table_latex, file=f)

print("\n")
print("------------ forward feature selection ---------------")
print(forward_table)
forward_table_latex = generate_latax_table(forward_table, caption="Results of Forward Feature Selection for different features", label="forward")
with open(os.path.join("tables", "forward_table_latex.txt"), 'w') as f:
    print(forward_table_latex, file=f)


print("\n\n")
print("----------------------Important! READ -------------------")
print("images are saved in images folder")
print("latex code for tables are saved in tables folder")
print("predictions are saved in results.csv file")

# Visualize the model and save it on mlp_structure image
# dummy_input = torch.randn(1, input_size)
# from torchviz import make_dot
# dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
# dot.render("mlp_structure", format="png", cleanup=True)

