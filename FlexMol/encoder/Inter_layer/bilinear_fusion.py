import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import InteractionLayer

class BilinearFusion(InteractionLayer):
    """
    Bilinear Fusion Layer for combining features from two different sources.
    """
    def __init__(self, feature_dim1, feature_dim2, output_dim=128):
        super(BilinearFusion, self).__init__()
        self.bilinear = nn.Bilinear(feature_dim1, feature_dim2, output_dim)
        self.activation = nn.ReLU()
        self.output_dim  = output_dim
    
    def get_output_shape(self):
        return self.output_dim

    def forward(self, features1, features2):
        """
        Forward pass for bilinear fusion.

        Args:
            features1 (Tensor): Features from the first source of shape (batch_size, feature_dim1).
            features2 (Tensor): Features from the second source of shape (batch_size, feature_dim2).

        Returns:
            Tensor: Fused features of shape (batch_size, output_dim).
        """
        fused_features = self.bilinear(features1, features2)
        fused_features = self.activation(fused_features)
        return fused_features