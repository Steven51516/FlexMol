import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from .base import * 

__all__ = ['DGL_NeuralFP']

# adapted from https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/nf.py

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_NeuralFP(EncoderLayer):
    """
    DGL_NeuralFP is a graph neural network implementation using DGL's NFGNN.
    """

    def __init__(self, in_feats=74, hidden_feats=[64, 64, 64], max_degree=10, activation=[F.relu, F.relu, F.relu], predictor_hidden_size=128, predictor_activation=torch.tanh, predictor_dim=128, device='cpu', readout=True):
        """
        Initialize the DGL_NeuralFP model.

        Parameters:
            in_feats (int): Number of input features.
            hidden_feats (list): List of hidden feature sizes for each GNN layer.
            max_degree (int): Maximum degree of the nodes.
            activation (list): List of activation functions for each GNN layer.
            predictor_hidden_size (int): Size of the hidden layer in the predictor.
            predictor_activation (function): Activation function for the predictor.
            predictor_dim (int): Dimension of the predictor.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
            readout (bool): Whether to use a readout layer.
        """
        super(DGL_NeuralFP, self).__init__()
        from dgllife.model.gnn.nf import NFGNN
        from dgllife.model.readout.sum_and_max import SumAndMax
        self.device = device
        
        self.gnn = NFGNN(in_feats=in_feats,
                         hidden_feats=hidden_feats,
                         max_degree=max_degree,
                         activation=activation)
        
        gnn_out_feats = self.gnn.gnn_layers[-1].out_feats
        self.node_to_graph = nn.Linear(gnn_out_feats, predictor_hidden_size)
        self.predictor_activation = predictor_activation

        if readout:
            self.readout = SumAndMax()
            self.output_shape = predictor_dim
            self.transform = nn.Linear(predictor_hidden_size * 2, predictor_dim)
        else:
            self.readout = None
            self.output_shape = (max_degree, predictor_dim)
            self.transform = nn.Linear(predictor_hidden_size, predictor_dim)

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
        feats = bg.ndata.pop('h')
        node_feats = self.gnn(bg, feats)
        node_feats = self.node_to_graph(node_feats)
        if self.readout:
            graph_feats = self.readout(bg, node_feats)
            graph_feats = self.predictor_activation(graph_feats)
            return self.transform(graph_feats)
        else:
            batch_size = bg.batch_size
            node_feats = self.transform(node_feats)
            return node_feats.view(batch_size, -1, self.output_shape[1])
