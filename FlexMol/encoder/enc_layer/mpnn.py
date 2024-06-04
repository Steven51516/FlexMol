import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from .base import *
from dgl.nn.pytorch import WeightAndSum
from dgllife.model.gnn import MPNNGNN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

__all__ = ['DGL_MPNN']

#adapted from https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/mpnn.py

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_MPNN(EncoderLayer):
    """
    DGL_MPNN is a graph neural network implementation using DGL's MPNNGNN.
    """

    def __init__(self, node_in_feats=15, edge_in_feats=5, node_out_feats=64, edge_hidden_feats=128, num_step_message_passing=6, output_feats=64, device='cpu'):
        """
        Initialize the DGL_MPNN model.

        Parameters:
            node_in_feats (int): Size of the input node features.
            edge_in_feats (int): Size of the input edge features.
            node_out_feats (int): Size of the output node representations.
            edge_hidden_feats (int): Size of the hidden edge representations.
            num_step_message_passing (int): Number of message passing steps.
            output_feats (int): Number of output features.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        super(DGL_MPNN, self).__init__()
        self.device = device
        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           edge_in_feats=edge_in_feats,
                           node_out_feats=node_out_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        gnn_out_feats = node_out_feats
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.output_shape = output_feats
        self.transform = nn.Linear(gnn_out_feats * 2, output_feats)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, bg):
        bg = bg.to(self.device)
        node_feats = bg.ndata.pop('n_feat')
        edge_feats = bg.edata.pop('e_feat')
        node_feats = self.gnn(bg, node_feats, edge_feats)
        return self.transform(self.readout(bg, node_feats))

