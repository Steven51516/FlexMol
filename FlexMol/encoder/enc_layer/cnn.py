import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import *

__all__ = ['CNN']

#adapted from https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/encoders.py

@register_to_device(True)
@register_loadtime_transform(True)
class CNN(EncoderLayer):
    """
    CNN is a convolutional neural network implementation for sequence-based data.
    """

    def __init__(self, in_channel, max_seq, filters=[32, 64, 96], output_feats=256, kernels=[4, 6, 8], device='cpu'):
        """
        Initialize the CNN model.

        Parameters:
            in_channel (int): Number of input channels.
            max_seq (int): Maximum sequence length.
            filters (list): List of filter sizes for each convolutional layer.
            output_feats (int): Number of output features.
            kernels (list): List of kernel sizes for each convolutional layer.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        super(CNN, self).__init__()
        self.device = device
        input_size = (in_channel, max_seq)
        channels = [in_channel] + filters
        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernels[i])
            for i in range(len(filters))
        ])
        n_size = self._conv_output_size(input_size)
        self.fc = nn.Linear(n_size, output_feats)
        self.output_feats = output_feats

    def get_output_shape(self):
        """
        Return the output shape of the model.

        Returns:
            int: Output shape of the model.
        """
        return self.output_feats
    
    @staticmethod
    def default_config(task, method):
        """
        Return the default configuration settings for the CNN.

        Parameters:
            task (str): Task name, e.g., 'drug' or 'prot_seq'.
            method (str): Method name.

        Returns:
            dict: Default configuration settings.
        """
        config = {}
        if task == "drug":
            config = {
                "in_channel": 63, 
                "max_seq": 100
            }
        elif task == "prot_seq":
            config = {
                "in_channel": 26, 
                "max_seq": 1000
            }
        return config

    def _conv_output_size(self, shape):
        """
        Compute the output size after the convolutional layers.

        Parameters:
            shape (tuple): Shape of the input tensor.

        Returns:
            int: Output size after the convolutional layers.
        """
        input_tensor = torch.rand(1, *shape)
        output_feat = self._forward_conv(input_tensor)
        n_size = output_feat.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        """
        Forward pass through the convolutional layers.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the convolutional layers.
        """
        for conv in self.conv:
            x = F.relu(conv(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output features after passing through the convolutional and fully connected layers.
        """
        x = x.to(self.device).float()
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
