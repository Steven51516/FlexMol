import torch
import torch.nn as nn
from .base import InteractionLayer

class Highway(InteractionLayer):
    r"""Highway network layer.

    Args:
        num_highway_layers (int): Number of highway layers.
        input_size (int): Size of the input and output for each layer.
        activation_fn (callable, optional): The non-linear activation function to use.

    Attributes:
        non_linear (ModuleList): List of linear layers for the T-transform.
        linear (ModuleList): List of linear layers for the carry gate.
        gate (ModuleList): List of linear layers for the transform gate.
        act (Activation): Activation function after non-linear transformation.
    """

    def __init__(self, input_size, num_highway_layers=1, activation_fn=nn.ReLU()):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_highway_layers)])
        self.act = activation_fn
        self.input_size = input_size


    def get_output_shape(self):
        return self.input_size
         
    def forward(self, x):
        r"""Forward pass of the highway layers.

        Args:
            x (Tensor): The input tensor to the highway layer.

        Returns:
            Tensor: The output tensor from the highway layer.
        """
        for i in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[i](x))
            non_linear = self.act(self.non_linear[i](x))
            linear = self.linear[i](x)
            x = gate * non_linear + (1 - gate) * linear

        return x

