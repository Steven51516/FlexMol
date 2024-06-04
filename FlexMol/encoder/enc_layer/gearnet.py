from collections.abc import Sequence

import torch
from torch import nn
from torch_scatter import scatter_add
from .base import *
from torchdrug import core, layers
from torchdrug.core import Registry as R
from torchdrug import data
from collections import deque
from collections.abc import Mapping, Sequence
#This collate_func is only used for GeometryAwareRelationalGraphNeuralNetwork

def collate_func(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    # elif isinstance(elem, Mapping):
    #     return {key: graph_collate([d[key] for d in batch]) for key in elem}
    # elif isinstance(elem, Sequence):
    #     it = iter(batch)
    #     elem_size = len(next(it))
    #     if not all(len(elem) == elem_size for elem in it):
    #         raise RuntimeError('Each element in list of batch should be of equal size')
    #     return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))


@register_to_device(True)
class GeometryAwareRelationalGraphNeuralNetwork(EncoderLayer):
    """
    Geometry Aware Relational Graph Neural Network proposed in
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_angle_bin (int, optional): number of bins to discretize angles between edges.
            The discretized angles are used as relations in edge message passing.
            If not provided, edge message passing is disabled.
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum", device = 'cpu'):
        super(GeometryAwareRelationalGraphNeuralNetwork, self).__init__()

        self.device = device
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)
        self._training_setup['collate_func'] = collate_func
    def get_output_shape(self):
        return self.output_dim

    @staticmethod
    def default_config(task, method):
        config_map = {
            "prot_3d": {
                "GearNet": {
                'input_dim': 21,
                'hidden_dims': [512, 512, 512, 512, 512, 512],
                'batch_norm': True,
                'concat_hidden': True,
                'short_cut': True,
                'readout': 'sum',
                'num_relation': 7,
                'edge_input_dim':59,
                'num_angle_bin':8,
              }
            }
        }

        task_config = config_map.get(task, {})
        return task_config.get(method, {})

    def forward(self, graph, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        from torchdrug.layers.geometry.function import AlphaCarbonNode,SequentialEdge,SpatialEdge,KNNEdge
        from torchdrug.layers.geometry.graph import GraphConstruction
        node_layers = [AlphaCarbonNode()]
        edge_layers = [SequentialEdge(max_distance = 2), SpatialEdge(radius =10.0,min_distance=5), KNNEdge(k=10,min_distance = 5)]
        edge_feature = 'gearnet'
        constract = GraphConstruction(node_layers=node_layers,edge_layers=edge_layers,edge_feature=edge_feature)
        graph = constract(graph)
        hiddens = []
        graph = graph.to(self.device)
        input = graph.node_feature.float()
        layer_input = input
        graph = graph.to(self.device)
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)
        return graph_feature
        #return {
        #    "graph_feature": graph_feature,
        #    "node_feature": node_feature
        #}