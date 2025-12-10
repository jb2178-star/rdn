# neural_net.py
import torch
from torch import nn

class NeuralNetModel1(nn.Module):
    def __init__(self, in_dim: int = 9, hidden_dim: int = 256, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), #outputs relative price C/S                               
        )

    def forward(self, x):
        #x: (batch_size, 9)
        return self.net(x)