import matplotlib.pyplot as plt


# Get the weights of a specific layer (change 'layer_name' to the name of your layer)
layer_name = 'fc2'
layer = dict(model.named_children())[layer_name]
weights = layer.weight.data.cpu().numpy()

# Visualize the weights
plt.figure(figsize=(10, 5))
plt.imshow(weights, cmap='viridis', aspect='auto')
plt.title(f'Weights of {layer_name}')
plt.colorbar()
plt.show()
