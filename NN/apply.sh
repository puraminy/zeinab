 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/NN/models.py b/NN/models.py
index 588de588edbe76b37d90ffcdca524a035f563dab..2a71bc9d4ae9d41c1c5cbd76c987a3c705952699 100644
--- a/NN/models.py
+++ b/NN/models.py
@@ -1,98 +1,100 @@
 import torch
 import torch.nn as nn
 
 ##################### New Models
 class RBFN(nn.Module):
-    def __init__(self, input_size, hidden_sizes):
+    def __init__(self, input_size, hidden_sizes, output_size=1):
         super(RBFN, self).__init__()
         hidden_size = hidden_sizes[0]
         self.hidden = nn.Linear(input_size, hidden_size)
-        self.output = nn.Linear(hidden_size, 1)
+        self.output = nn.Linear(hidden_size, output_size)
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
-    def __init__(self, input_size, sigma=1.0):
+    def __init__(self, input_size, sigma=1.0, output_size=1):
         super(GRNN, self).__init__()
         self.pattern_layer = nn.Linear(input_size, input_size, bias=False)
+        self.output_layer = nn.Linear(1, output_size)
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
-        output = (basis_functions @ self.pattern_layer.weight).sum(dim=1, keepdim=True) / summation_layer
+        weighted = (basis_functions @ self.pattern_layer.weight).sum(dim=1, keepdim=True) / summation_layer
+        output = self.output_layer(weighted)
         return output
 
 
 #class CNN(nn.Module):
 #    def __init__(self, input_channels, num_classes):
 #        super(CNN, self).__init__()
 #        self.layer1 = nn.Sequential(
 #            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
 #            nn.BatchNorm2d(16),
 #            nn.ReLU(),
 #            nn.MaxPool2d(kernel_size=2, stride=2))
 #        self.layer2 = nn.Sequential(
 #            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
 #            nn.BatchNorm2d(32),
 #            nn.ReLU(),
 #            nn.MaxPool2d(kernel_size=2, stride=2))
 #        self.fc1 = nn.Linear(32 * 7 * 7, 1000)
 #        self.fc2 = nn.Linear(1000, num_classes)
 #
 #    def forward(self, x):
 #        out = self.layer1(x)
 #        out = self.layer2(out)
 #        out = out.view(out.size(0), -1)
 #        out = self.fc1(out)
 #        out = self.fc2(out)
 #        return out
 #
 #
 class Linear1HiddenLayer(nn.Module):
     activation1 = None
-    def __init__(self, input_size, hidden_sizes, output_zide=1):
+    def __init__(self, input_size, hidden_sizes, output_size=1):
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
         self.hidden1_to_output = nn.Linear(hidden_sizes[0], output_size)
 
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
 
EOF
)