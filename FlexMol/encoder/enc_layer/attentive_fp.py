import torch
import torch.nn as nn
import dgl
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from .base import *

__all__ = ['DGL_AttentiveFP']

#adapted from https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/attentivefp.py

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_AttentiveFP(EncoderLayer):
    """
    DGL_AttentiveFP is a graph neural network implementation using DGL's AttentiveFPGNN and AttentiveFPReadout.
    """

    def __init__(self, node_feat_size=39, edge_feat_size=11, num_layers=2, num_timesteps=2, graph_feat_size=200, predictor_dim=64, device='cpu'):
        """
        Initialize the DGL_AttentiveFP model.

        Parameters:
            node_feat_size (int): Size of the node features.
            edge_feat_size (int): Size of the edge features.
            num_layers (int): Number of GNN layers.
            num_timesteps (int): Number of readout timesteps.
            graph_feat_size (int): Size of the graph features.
            predictor_dim (int): Dimension of the predictor.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        super(DGL_AttentiveFP, self).__init__()
        self.device = device
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size)

        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps)

        self.transform = nn.Linear(graph_feat_size, predictor_dim)
        self.output_shape = predictor_dim

    def get_output_shape(self):
        """
        Return the output shape of the model.

        Returns:
            int: Output shape of the model.
        """
        return self.output_shape

    def forward(self, bg):
        """
        Forward pass of the model.

        Parameters:
            bg (DGLGraph): Batched DGL graph.

        Returns:
            torch.Tensor: Output features after passing through the GNN and readout layers.
        """
        bg = bg.to(self.device)
        node_feats = bg.ndata.pop('h')
        edge_feats = bg.edata.pop('e')
        node_feats = self.gnn(bg, node_feats, edge_feats)
        graph_feats = self.readout(bg, node_feats, False)
        return self.transform(graph_feats)
