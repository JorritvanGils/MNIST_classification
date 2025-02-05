import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network.nn import MLP

# Initialize the neural network
n = MLP(3, [4, 4, 1])  
# print("Initial model parameters:")

# # Print all parameters before training
# for i, p in enumerate(n.parameters()):
#     print(f"Param {i}: {p.data}")

# Training data
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # Target labels



def train_model(model, inputs, targets, epochs=100, lr=0.1):
    """Trains a model using basic gradient descent.

    Args:
        model (torch.nn.Module): The neural network model.
        inputs (list): List of input tensors.
        targets (list): List of target values.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 0.1.
    """
    print("Training starts...")

    final_loss = None 

    for epoch in range(epochs):
        # Forward pass
        predictions = [model(x) for x in inputs]
        loss = sum((pred - target) ** 2 for pred, target in zip(predictions, targets))

        # Zero gradients before backpropagation
        for param in model.parameters():
            param.grad = 0.0
        loss.backward()

        # Parameter update
        for param in model.parameters():
            param.data -= lr * param.grad

        final_loss = loss.data

        print(f"Epoch {epoch+1}: Loss = {loss.data}")

    print(f"\nFinal Loss after {epochs} epochs: {final_loss}")

    print("\nFinal Predictions After Training:")
    for i, x in enumerate(inputs):
        print(f"Input {i}: {x}, Prediction: {model(x).data}, Target: {targets[i]}")


# train_model(n, xs, ys, epochs=100, lr=0.1) # loss 4.0
# train_model(n, xs, ys, epochs=100, lr=0.01) 
# train_model(n, xs, ys, epochs=100, lr=0.05) 
train_model(n, xs, ys, epochs=100, lr=0.04) 
