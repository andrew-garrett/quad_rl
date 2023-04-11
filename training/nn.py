import torch
from torch import nn
from torchvision import MLP

activations = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid
}

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = self.config["model"]["state_dim"] + self.config["model"]["control_dim"]
        self.output_size = self.config["model"]["state_dim"]
        self.layers = self.config["model"]["hidden_layers"] + [self.output_size]

        self.net = MLP(
            in_channels = self.input_size,
            hidden_channels = self.layers,
            norm_layer = nn.BatchNorm1d,
            activation_layer = activations[self.config["model"]["activation"]],
            dropout = self.config["model"]["dropout"]
        )
    
    def forward(self, state, control):
        mlp_input = torch.cat([state, control], dim=-1)
        mlp_output = self.net(mlp_input)
        return mlp_output