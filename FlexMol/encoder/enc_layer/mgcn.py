from dgl.nn.pytorch import WeightAndSum
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from .base import *
from dgllife.model.gnn import MGCNGNN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

__all__ = ['DGL_MGCN']

#adapted from https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/mgcn.py

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_MGCN(EncoderLayer):
    """
    DGL_MGCN is a graph neural network implementation using DGL's MGCNGNN.
    """

    def __init__(self, feats=128, n_layers=3, num_node_types=100,
                 num_edge_types=3000, cutoff=30., gap=0.1, output_feats=64, device='cpu'):
        """
        Initialize the DGL_MGCN model.

        Parameters:
            feats (int): Number of features.
            n_layers (int): Number of GNN layers.
            num_node_types (int): Number of node types.
            num_edge_types (int): Number of edge types.
            cutoff (float): Cutoff distance for edges.
            gap (float): Gap parameter for distance bins.
            output_feats (int): Number of output features.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        super(DGL_MGCN, self).__init__()
        self.device = device
        self.gnn = MGCNGNN(feats=feats,
                           n_layers=n_layers,
                           num_node_types=num_node_types,
                           num_edge_types=num_edge_types,
                           cutoff=cutoff,
                           gap=gap)
        gnn_out_feats = (n_layers + 1) * feats
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.output_shape = output_feats
        self.transform = nn.Linear(gnn_out_feats * 2, output_feats)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, bg):
        bg = bg.to(self.device)
        node_types = bg.ndata.pop('node_type')
        edge_distances = bg.edata.pop('distance')
        node_feats = self.gnn(bg, node_types, edge_distances)
        return self.transform(self.readout(bg, node_feats))
