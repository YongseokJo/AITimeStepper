import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu1 = nn.ReLU()
        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()
        #self.relu2 = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        # Forward pass: input -> fc1 -> ReLU -> fc2
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        #x = self.sigmoid(x)
        #x = (torch.tanh(x) + 1) / 2
        #x = self.relu2(x)
        x = self.softplus(x)
        return x 



class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], activation='relu', dropout=0.0, output_positive=True):
        """
        Initializes the fully connected network.

        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dims (list of int): Sizes of hidden layers. Default is [64, 64].
            activation (str): Activation function to use. Options: 'relu', 'tanh', or 'sigmoid'. Default is 'relu'.
            dropout (float): Dropout probability (0.0 means no dropout). Default is 0.0.
        """
        super(FullyConnectedNN, self).__init__()

        # Map string to actual activation class
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'silu': nn.SiLU
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from 'relu', 'tanh', or 'sigmoid'.")
        act_fn = activations[activation.lower()]

        layers = []
        current_dim = input_dim

        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = h

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # Append Softplus activation to force outputs to be positive, if desired
        if output_positive:
            layers.append(nn.Softplus())

        # Combine layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        #Initializes the weights of Linear layers.
        #Uses Kaiming initialization for ReLU activation and Xavier for others.
        #Biases are initialized to zero.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation == 'relu':
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

