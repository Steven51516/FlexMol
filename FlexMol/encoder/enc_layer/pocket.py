import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import TAGConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
from .base import *

__all__ = ['TAG']

@register_collate_func(dgl.batch)
@register_to_device(True)
class TAG(EncoderLayer):
    """
    TAG is a graph neural network implementation using DGL's TAGConv.
    """

    def __init__(self, device='cpu', pocket_num=30, output_feats=128, pooling=True):
        """
        Initialize the TAG model.

        Parameters:
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
            pocket_num (int): Number of pockets in the graph.
            output_feats (int): Number of output features.
            pooling (bool): Whether to use pooling.
        """
        super(TAG, self).__init__()
        self.protein_graph_conv = nn.ModuleList()
        self.protein_graph_conv.append(TAGConv(30, 64, 2))
        for i in range(3):
            self.protein_graph_conv.append(TAGConv(64, 64, 2))

        self.pooling_ligand = nn.Linear(64, 1)
        self.pooling_protein = nn.Linear(64, 1)

        self.dropout = 0.2
        self.output_shape = output_feats
        self.device = device
        self.pocket_num = pocket_num
        self.transform = nn.Linear(128, output_feats)
        self.pooling = pooling
        if pooling:
            self.output_shape = output_feats
        else:
            self.output_shape = (pocket_num, output_feats)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, g):
        """
        Forward pass of the model.

        Parameters:
            g (DGLGraph): Batched DGL graph.

        Returns:
            torch.Tensor: Output features after passing through the GNN and pooling layers.
        """
        g = g.to(self.device)
        feature_protein = g.ndata['h']
        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g, feature_protein))
        
        max_pool = MaxPooling()
        max_pooled_protein_rep = max_pool(g, feature_protein)
    
        attn_pool_protein = GlobalAttentionPooling(self.pooling_protein)
        attn_pooled_protein_rep = attn_pool_protein(g, feature_protein).view(-1, 64)
        protein_rep = torch.cat([max_pooled_protein_rep, attn_pooled_protein_rep], dim=1)
        protein_rep = self.transform(protein_rep)
    
        B = len(protein_rep) // self.pocket_num
        protein_rep_reshaped = protein_rep.view(B, self.pocket_num, -1)
        if self.pooling:
            protein_rep_reshaped = torch.mean(protein_rep_reshaped, dim=1)
        return protein_rep_reshaped
