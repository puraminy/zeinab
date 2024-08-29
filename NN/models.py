import torch
import torch.nn as nn

##################### New Models
class RBFN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(RBFN, self).__init__()
        hidden_size = hidden_sizes[0]
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(self.hidden)
        # self.hidden_layers.append(self.output)

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
        # self.hidden_layers.append(self.hidden1_to_output)

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
        self.hidden1_to_hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[-1])
        self.hidden2_to_output = nn.Linear(hidden_sizes[-1], 1)

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(self.input_to_hidden1)
        self.hidden_layers.append(self.hidden1_to_hidden2)
        # self.hidden_layers.append(self.hidden2_to_output)

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

class FFNN(nn.Module):
    activations = [nn.ReLU(), nn.ReLU(), nn.ReLU()]  # Specify activations for hidden layers
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        # self.activations = activations if activations is not None else []

        # Input to first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Intermediate hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Last hidden layer to output
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            act = self.activations[-1]
            if i < len(self.activations) and self.activations[i] is not None:
               act = self.activations[i]
            if act is not None:
               x = act(x)
        x = self.output_layer(x)
        return x

class ReluFFNN(FFNN):
      activations = [nn.ReLU()]  # Specify activations for hidden layers

class LinearFFNN(FFNN):
      activations = [None]  # Specify activations for hidden layers

class TanhFFNN(FFNN):
      activations = [nn.Tanh()]  # Specify activations for hidden layers

class TanhReluFFNN(FFNN):
      activations = [nn.Tanh(), nn.ReLU()]  # Specify activations for hidden layers

class ReluTanhFFNN(FFNN):
      activations = [nn.ReLU(), nn.Tanh()]  # Specify activations for hidden layers

class TanhLinFFNN(FFNN):
      activations = [nn.Tanh(), nn.Identity()]  # Specify activations for hidden layers

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

# LSTM Model (Still included for comparison)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes): # in ezafeh shod va ziri hazf shod
    # def __init__(self, input_size, hidden_size, num_layers, output_size, transfer_function):
        super(LSTM, self).__init__()
        # be jaaye parametrhaye dadeh shode inha estefadeh shod
        num_layers = 2
        hidden_size = hidden_sizes[0]
        output_size = hidden_sizes[1] if len(hidden_sizes) > 1 else 1
        transfer_function = nn.ReLU

        ###########  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # hame modelha in hidden_layers ro daran
        self.hidden_layers = [self.lstm, self.fc]

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.transfer_function = transfer_function

    def forward(self, x):
        print("Input shape:", x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        print("LSTM output shape:", out.shape)
        out = self.fc(out[:, -1, :])
        print("FC output shape:", out.shape)

        if out.numel() > 0:
            out = self.transfer_function()(out)
        return out
