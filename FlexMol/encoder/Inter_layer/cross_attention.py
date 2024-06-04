import torch
import torch.nn as nn
from .base import InteractionLayer


class BidirectionalCrossAttention(InteractionLayer):
    """
    A bidirectional cross-attention module with independent max pooling over attended outputs,
    designed to handle variable sequence lengths between two embedding sets.
    """
    def __init__(self, embed_dim1, embed_dim2, common_dim=128, num_heads=4, use_residual=True):
        super(BidirectionalCrossAttention, self).__init__()
        self.proj1 = nn.Linear(embed_dim1, common_dim)
        self.proj2 = nn.Linear(embed_dim2, common_dim)
        self.output_shape = common_dim * 2
        self.mix_attn = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads)
        self.use_residual = use_residual
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.common_dim = common_dim

    def get_output_shape(self):
        return self.output_shape

    def forward(self, embeddings1, embeddings2):

        embeddings1 = self.proj1(embeddings1)
        embeddings2 = self.proj2(embeddings2)


        transposed1 = embeddings1.transpose(0, 1)
        transposed2 = embeddings2.transpose(0, 1)


        attended1, _ = self.mix_attn(query=transposed1, key=transposed2, value=transposed2)
        attended2, _ = self.mix_attn(query=transposed2, key=transposed1, value=transposed1)

        if self.use_residual:
            attended1 += transposed1
            attended2 += transposed2

        attended1 = attended1.transpose(0, 1)
        attended2 = attended2.transpose(0, 1)

        pooled1 = self.max_pool(attended1.permute(0, 2, 1)).squeeze(-1)
        pooled2 = self.max_pool(attended2.permute(0, 2, 1)).squeeze(-1)

        concatenated_output = torch.cat((pooled1, pooled2), dim=1)

        return concatenated_output





