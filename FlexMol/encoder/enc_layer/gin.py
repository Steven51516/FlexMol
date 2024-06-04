import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GINConv
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from .base import *

__all__ = ['DGL_GIN']

#adapted from https://github.com/dmlc/dgl/blob/master/examples/mxnet/gin/gin.py

class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linears[0](x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_GIN(EncoderLayer):
    """
    DGL_GIN is a graph isomorphism network implementation using DGL's GINConv.
    """

    def __init__(self, in_feats=74, hidden_dim=64, output_dim=64, num_layers=5, num_mlp_layers=2, final_dropout=0.5, learn_eps=True, neighbor_pooling_type="sum", device='cpu', max_nodes=50, readout=True):
        """
        Initialize the DGL_GIN model.

        Parameters:
            in_feats (int): Number of input features.
            hidden_dim (int): Number of hidden features.
            output_dim (int): Number of output features.
            num_layers (int): Number of GIN layers.
            num_mlp_layers (int): Number of MLP layers in each GIN layer.
            final_dropout (float): Dropout rate for the final fully connected layer.
            learn_eps (bool): Whether to learn the epsilon weighting.
            neighbor_pooling_type (str): Type of neighbor pooling ('sum', 'mean', 'max').
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
            max_nodes (int): Maximum number of nodes in the graph.
            readout (bool): Whether to use a readout layer.
        """
        super(DGL_GIN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers - 1):
            if i == 0:
                mlp = MLP(num_mlp_layers, in_feats, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.linears_prediction.append(nn.Linear(in_feats, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)
        gnn_out_feats = hidden_dim
        from dgl.nn.pytorch.glob import AvgPooling
        if readout:
            self.readout = AvgPooling()
            self.output_shape = output_dim
            self.transform = nn.Linear(gnn_out_feats, output_dim)
        else:
            self.readout = None
            self.output_shape = (max_nodes, output_dim)
            self.transform = nn.Linear(gnn_out_feats, output_dim)

    def get_output_shape(self):
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        config_map = {
            "drug": {
                "GIN": {"in_feats": 74}
            },
            "prot_3d": {
                "GIN": {"in_feats": 25},
                "GIN_ESM": {"in_feats": 1305}
            }
        }
        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, bg):
        bg = bg.to(self.device)
        feats = bg.ndata.pop('h').float()

        h = feats
        hidden_rep = [h]
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](bg, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        if self.readout:
            bg.ndata['h'] = h
            return self.transform(self.readout(bg, h))
        else:
            batch_size = bg.batch_size
            node_feats = self.transform(h)
            return node_feats.view(batch_size, -1, self.output_shape[1])
