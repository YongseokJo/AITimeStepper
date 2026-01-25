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



class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, scale: float):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.scale = scale
        # Random Gaussian matrix B: (embedding_dim // 2, input_dim)
        # We divide by 2 because we will concat sin and cos, doubling the size.
        self.B = nn.Parameter(torch.randn(embedding_dim // 2, input_dim) * scale, requires_grad=False)

    def forward(self, x):
        # x: (..., input_dim)
        # proj: (..., embedding_dim // 2)
        proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FullyConnectedNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[64, 64],
        activation='relu',
        dropout=0.0,
        output_positive=True,
        fourier_scale=-1.0,
        fourier_dim=256,
    ):
        """
        Initializes the fully connected network with optional Fourier Features.
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
        
        # Fourier Embedding Layer
        if fourier_scale > 0:
            self.embedding = FourierFeatureEmbedding(input_dim, fourier_dim, fourier_scale)
            current_dim = fourier_dim
        else:
            self.embedding = None
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
        if self.embedding is not None:
            x = self.embedding(x)
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

