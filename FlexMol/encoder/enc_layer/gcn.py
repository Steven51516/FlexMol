import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from .base import *
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

__all__ = ['DGL_GCN']

#adapted from https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/gcn.py

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_GCN(EncoderLayer):
    """
    DGL_GCN is a graph convolutional network implementation using DGL's GCN and WeightedSumAndMax.
    """

    def __init__(self, in_feats=74, hidden_feats=[64, 64, 64], activation=[F.relu, F.relu, F.relu], output_feats=64, device='cpu', max_nodes=50, readout=True):
        """
        Initialize the DGL_GCN model.

        Parameters:
            in_feats (int): Number of input features.
            hidden_feats (list): List of hidden feature sizes for each GCN layer.
            activation (list): List of activation functions for each GCN layer.
            output_feats (int): Number of output features.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
            max_nodes (int): Maximum number of nodes in the graph.
            readout (bool): Whether to use a readout layer.
        """
        super(DGL_GCN, self).__init__()
        self.device = device
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=activation)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        if readout:
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.output_shape = output_feats
            self.transform = nn.Linear(gnn_out_feats * 2, output_feats)
        else:
            self.readout = None
            self.output_shape = (max_nodes, output_feats)
            self.transform = nn.Linear(gnn_out_feats, output_feats)

    def get_output_shape(self):
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        config_map = {
            "drug": {
                "GCN": {"in_feats": 74}
            },
            "prot_3d": {
                "GCN": {"in_feats": 25},
                "GCN_ESM": {"in_feats": 1305}
            }
        }
    
        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, bg):
        bg = bg.to(self.device)
        feats = bg.ndata.pop('h')
        feats = feats.to(torch.float32)
        node_feats = self.gnn(bg, feats)
        if self.readout:
            return self.transform(self.readout(bg, node_feats))
        else:
            batch_size = bg.batch_size
            node_feats = self.transform(node_feats)
            return node_feats.view(batch_size, -1, self.output_shape[1])
