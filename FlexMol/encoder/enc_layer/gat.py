import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F
from .base import *
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

__all__ = ['DGL_GAT']

@register_collate_func(dgl.batch)
@register_to_device(True)
class DGL_GAT(EncoderLayer):
    """
    DGL_GAT is a graph attention network implementation using DGL's GATConv layers.
    """

    def __init__(self, in_feats=74, num_heads=[2, 2, 2], hidden_feats=[64, 64, 64], activation=[F.relu, F.relu, F.relu], output_feats=64, feat_drop=0.1, attn_drop=0.1, device='cpu', max_nodes=50, readout=True):
        """
        Initialize the DGL_GAT model.

        Parameters:
            in_feats (int): Number of input features.
            num_heads (list): Number of attention heads for each GAT layer.
            hidden_feats (list): Number of hidden features for each GAT layer.
            activation (list): Activation functions for each GAT layer.
            output_feats (int): Number of output features.
            feat_drop (float): Dropout rate for feature dropouts.
            attn_drop (float): Dropout rate for attention dropouts.
            device (str): Device to run the model on.
            max_nodes (int): Maximum number of nodes in the graph.
            readout (bool): Whether to use a readout layer.
        """
        super(DGL_GAT, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        assert len(hidden_feats) == len(num_heads) == len(activation), "Length of hidden_feats, num_heads, and activation must match"
        self.num_heads = num_heads
        current_dim = in_feats
        for i in range(len(hidden_feats)):
            self.layers.append(
                GATConv(
                    in_feats=current_dim,
                    out_feats=hidden_feats[i],
                    num_heads=num_heads[i],
                    activation=activation[i],
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                )
            )
            current_dim = hidden_feats[i] * num_heads[i]

        if readout:
            self.readout = WeightedSumAndMax(hidden_feats[-1])
            self.output_shape = output_feats
            self.transform = nn.Linear(hidden_feats[-1] * 2, output_feats)
        else:
            self.readout = None
            self.output_shape = (max_nodes, output_feats)
            self.transform = nn.Linear(hidden_feats[-1], output_feats)

    def get_output_shape(self):
        """
        Return the output shape of the model.

        Returns:
            int or tuple: Output shape of the model.
        """
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        """
        Return the default configuration settings for the encoder layer.

        Parameters:
            task (str): Task name, e.g., 'drug' or 'prot_3d'.
            method (str): Method name, e.g., 'GAT' or 'GAT_ESM'.

        Returns:
            dict: Default configuration settings.
        """
        config_map = {
            "drug": {
                "GAT": {"in_feats": 74}
            },
            "prot_3d": {
                "GAT": {"in_feats": 25},
                "GAT_ESM": {"in_feats": 1305}
            }
        }
    
        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, bg):
        """
        Forward pass of the model.

        Parameters:
            bg (DGLGraph): Batched DGL graph.

        Returns:
            torch.Tensor: Output features after passing through the GAT layers and optional readout.
        """
        bg = bg.to(self.device)
        h = bg.ndata.pop('h')
        h = h.to(torch.float32)
        for i, layer in enumerate(self.layers):
            h = layer(bg, h)
            if i < len(self.layers) - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)

        if self.readout:
            readout_result = self.readout(bg, h)
            transformed_result = self.transform(readout_result)
            return transformed_result
        else:
            batch_size = bg.batch_size
            transformed_feats = self.transform(h)
            return transformed_feats.view(batch_size, -1, self.output_shape[1])
