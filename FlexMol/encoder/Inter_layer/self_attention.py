import torch
import torch.nn as nn
from .base import InteractionLayer

class MultiHeadAttention(InteractionLayer):
    """
    Multi-head attention interaction layer. 

    :param embed_size: Size of each embedding vector.
    :param head_num: Number of attention heads.
    :param dropout: Dropout rate applied to the outputs of the attention mechanism.
    :param residual: Whether to include a residual connection around the attention.
    """
    def __init__(self, embed_size, head_num=4, dropout=0.1, residual=True):
        super(MultiHeadAttention, self).__init__()
        self.residual = residual
        self.attention = nn.MultiheadAttention(embed_size, head_num, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def get_output_shape(self):
        return "SAME_TO_PARENT"

    def forward(self, x):
        """
        Forward pass for the multi-head attention layer.

        :param x: Input tensor of shape (batch_size, feature_fields, embed_dim).
        :returns: Output tensor of the same shape as input, and attention weights.
        """
        x_norm = self.layer_norm(x)
        
        # Transpose x to fit the expected input shape for nn.MultiheadAttention (seq_len, batch_size, embed_dim)
        x_transposed = x_norm.transpose(0, 1)
        attn_output, attn_output_weights = self.attention(x_transposed, x_transposed, x_transposed)
        attn_output = attn_output.transpose(0, 1)
        
        if self.residual:
            attn_output = attn_output + x
        return attn_output