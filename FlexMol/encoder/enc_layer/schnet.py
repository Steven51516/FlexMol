import torch
import torch.nn as nn
import dgl
from .base import *
from dgllife.model.gnn import SchNetGNN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

__all__ = ['DGL_SchNet']

# adapted from https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/schnet.py

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_SchNet(EncoderLayer):
    """
    DGL_SchNet is a graph neural network implementation using DGL's SchNetGNN.
    """

    def __init__(self, node_feats=64, hidden_feats=[64, 64, 64], num_node_types=100, cutoff=30., gap=0.1, output_feats=64, device='cpu'):
        """
        Initialize the DGL_SchNet model.

        Parameters:
            node_feats (int): Number of node features.
            hidden_feats (list): List of hidden feature sizes for each GNN layer.
            num_node_types (int): Number of node types.
            cutoff (float): Cutoff distance for edges.
            gap (float): Gap parameter for distance bins.
            output_feats (int): Number of output features.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        super(DGL_SchNet, self).__init__()
        self.device = device
        self.gnn = SchNetGNN(node_feats, hidden_feats, num_node_types, cutoff, gap)
        gnn_out_feats = hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.output_shape = output_feats
        self.transform = nn.Linear(gnn_out_feats * 2, output_feats)

    def get_output_shape(self):
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
        node_types = bg.ndata.pop('node_type')
        edge_distances = bg.edata.pop('distance')
        node_feats = self.gnn(bg, node_types, edge_distances)
        return self.transform(self.readout(bg, node_feats))

