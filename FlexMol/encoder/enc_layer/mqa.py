import numpy as np
import torch
import torch.nn as nn
from FlexMol.util.biochem.protein.gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from .base import *

import dgl

#This collate_func is only used for MQAModel
def collate_func(batch):
    features_i0, features_i1, features_i2, features_i3 = [], [], [], []
    i4 = []
    batch_indices = []
    pre_node = 0
    
    for i, item in enumerate(batch):
        features_i0.append(item[0])
        features_i1.append(item[1])
        features_i2.append(item[2])
        features_i3.append(item[3])
        i4.append(item[4] + pre_node)
        pre_node += item[0].shape[0]
        batch_indices.extend([i] * item[0].shape[0])
    
    i0 = torch.cat(features_i0, dim=0)
    i1 = torch.cat(features_i1, dim=0)
    i2 = torch.cat(features_i2, dim=0)
    i3 = torch.cat(features_i3, dim=0)
    i4 = torch.cat(i4, dim=1)
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    return [i0, i1, i2, i3, i4, batch_indices]



# This MQAModel is copied from gvp-pytorch.model.
# Mainly modify the input of the forward function
@register_to_device(True)
class MQAModel(EncoderLayer):
    '''
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1, device = 'cpu'):
        self.device = device
        super(MQAModel, self).__init__()
        self.node_h_dim =node_h_dim
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
        '''
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, 1)
        )
        '''    
        self._training_setup['collate_func'] = collate_func
    def get_output_shape(self):
        return self.node_h_dim[0]
    @staticmethod
    def default_config(task, method):
        config_map = {
            "prot_3d": {
                "GVP": {"node_in_dim": (6,3), 
                             "node_h_dim": (100,16), 
                             "edge_in_dim":  (32, 1), 
                             "edge_h_dim":  (32, 1), 
                              }
            }
        }

        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self,input,seq = None): 
                    
        '''
         h_V, edge_index, h_E, seq=None, batch=None
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        h_V = (input[0].to(self.device),input[1].to(self.device))
        edge_index = input[4].to(self.device)
        h_E = (input[2].to(self.device),input[3].to(self.device))
        batch = input[5].to(self.device)

        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        
        if batch is None: out = out.mean(dim=0, keepdims=True)
        else: out = scatter_mean(out, batch, dim=0)
        
        return out #,self.dense(out).squeeze(-1) + 0.5