from .FM_config import *
import torch.nn as nn
import pickle
import hashlib
import torch
import os
import copy
from pathlib import Path


class NodeWrapper:

    def __init__(self, parents=[], children=[], model=None, root_idx=None,
                 output_shape=None):
        super(NodeWrapper, self).__init__()
        if not isinstance(parents, list):
            parents = [parents]
        if not isinstance(children, list):
            children = [children]
        self.parents = parents
        self.children = children
        self.model = model
        self.output_shape = output_shape
        self.root_idx = root_idx
        for parent in parents:
            parent.add_child(self)

    def add_parent(self, node):
        if node not in self.parents:
            self.parents.append(node)

    def add_child(self, node):
        if node not in self.children:
            self.children.append(node)

    def get_children(self):
        return self.children

    def get_parents(self):
        return self.parents

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return len(self.parents) == 0

    def get_model(self):
        return self.model

    def get_output_shape(self):
        return self.output_shape

    def get_root_idx(self):
        return self.root_idx


class EncoderNode:
    def __init__(self, method, pickle_dir=None, use_hash=False, **config):
        self.method = method
        self.featurizer = None
        self.model = None
        self.model_training_setup = None
        self.output_shape = None
        self.pickle_dir = pickle_dir
        self.use_hash = use_hash

        if 'data_dir' in config:
            config['data_dir'] = self.sanitize_path(config['data_dir'])

    def get_model(self):
        return self.model

    def get_featurizer(self):
        return self.featurizer

    def _generate_hash(self, text):
        """Generate a SHA-256 hash for the given text."""
        hash_object = hashlib.sha256(text.encode('utf-8'))
        return hash_object.hexdigest()

    def transform(self, entities, mode="default"):
        if self.pickle_dir is not None:
            transformed_entities = []
            for entity in entities:
                entity_id = self._generate_hash(entity) if self.use_hash else entity
                pickle_path = os.path.join(self.pickle_dir, f'{entity_id}.pkl')
                if os.path.exists(pickle_path):
                    with open(pickle_path, 'rb') as file:
                        transformed_entity = pickle.load(file)
                else:
                    temp = copy.deepcopy(entity)
                    transformed_entity = self.featurizer(temp, mode)
                    
                    if not os.path.exists(self.pickle_dir):
                        os.makedirs(self.pickle_dir)
                    with open(pickle_path, 'wb') as file:
                        pickle.dump(transformed_entity, file)

                transformed_entities.append(transformed_entity)

            return transformed_entities

        else:
            temp = copy.deepcopy(entities)
            return self.featurizer(temp, mode, batch = True)

    def get_output_shape(self):
        return self.model.get_output_shape()
    
    def sanitize_path(self, path):
        if not path.endswith('/'):
            path += '/'
        p = Path(path)
        if not p.exists() or not p.is_dir():
            raise ValueError(f"The path {path} does not exist or is not a directory.")
        return str(p)


class InteractionNode(nn.Module):
    def __init__(self, nodes, method, head=None, mlp_hidden_layers=None, **config):
        super().__init__()
        self.method = method
        parent_output_shapes = [node.get_output_shape() for node in nodes]
        parent_output_shapes = [(1, shape) if isinstance(shape, int) else shape for shape in parent_output_shapes]
        self.inter_layer, self.output_shape = init_inter_layer(method, parent_output_shapes, **config)
        self.mlp = self.setup_mlp(head, mlp_hidden_layers) if head else None


    def setup_mlp(self, head, mlp_hidden_layers):
        input_dim = self.output_shape
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [512, 264, 128]
        mlp_layers = []
        for i, hidden_dim in enumerate(mlp_hidden_layers):
            mlp_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(input_dim, head))
        return nn.Sequential(*mlp_layers)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, *encoded):
        out = torch.cat(encoded, dim=1) if self.method == "cat" else self.inter_layer(*encoded)
        return self.mlp(out) if self.mlp else out


class ConcatNode(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        parent_output_shapes = [node.get_output_shape() for node in nodes]
        are_all_integers = all(isinstance(shape, int) for shape in parent_output_shapes)
        if(are_all_integers):
            self.dim = 1
            self.output_shape = sum(parent_output_shapes)
        else:
            self.dim = 2
            parent_output_shapes = [(1, shape) if isinstance(shape, int) else shape for shape in parent_output_shapes]
            if not all(shape[0] == parent_output_shapes[0][0] for shape in parent_output_shapes):
                raise ValueError("All tensors must have the same size in the first dimension")
            self.output_shape = (parent_output_shapes[0][0], sum(shape[1] for shape in parent_output_shapes))

    def get_output_shape(self):
        return self.output_shape

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class StackNode(nn.Module):
    def __init__(self, nodes):
        super(StackNode, self).__init__()
        parent_output_shapes = [node.get_output_shape() for node in nodes]
        self.are_all_integers = all(isinstance(shape, int) for shape in parent_output_shapes)

        if self.are_all_integers:
            self.output_shape = (len(parent_output_shapes), parent_output_shapes[0])
        else:
            parent_output_shapes = [(1, shape) if isinstance(shape, int) else shape for shape in parent_output_shapes]
            if not all(shape[1] == parent_output_shapes[0][1] for shape in parent_output_shapes):
                raise ValueError("All tensors must have the same size in the first dimension")
            self.output_shape = (sum(shape[0] for shape in parent_output_shapes), parent_output_shapes[0][1])

    def get_output_shape(self):
        return self.output_shape

    def forward(self, *inputs):
        if self.are_all_integers:
            return torch.stack(inputs, dim=1)
        else:
            inputs = [input.unsqueeze(1) if input.dim() < max(input.dim() for input in inputs) else input for input in inputs]
            return torch.cat(inputs, dim=1)


class FlattenNode(nn.Module):
    def __init__(self, node):
        super().__init__()
        shape = node.get_output_shape()
        assert isinstance(shape, tuple), "Expected the shape to be a tuple"
        # Flatten the tensor dimensions into one dimension, assuming the shape is 2D (e.g., (dim1, dim2))
        self.output_shape = shape[0] * shape[1]


    def forward(self, input):
        batch_size, dim1, dim2 = input.size()
        return input.reshape(batch_size, -1)

    def get_output_shape(self):
        return self.output_shape



class MLPNode(nn.Module):
    def __init__(self, node, head, hidden_layers=None):
        super().__init__()
        self.input_shape = node.get_output_shape()
        if(isinstance(self.input_shape, tuple)):
            self.input_shape = self.input_shape[1]
        self.model = self.setup_mlp(head, hidden_layers)
        self.output_shape = head

    def setup_mlp(self, head, mlp_hidden_layers):
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [128, 64]
        mlp_layers = []
        input_dim = self.input_shape
        for hidden_dim in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(input_dim, head))
        return nn.Sequential(*mlp_layers)

    def forward(self, input):
        return self.model(input)

    def get_output_shape(self):
        return self.output_shape



class PoolingNode(nn.Module):
    def __init__(self, node, mode='max'):
        super(PoolingNode, self).__init__()
        self.input_shape = node.get_output_shape()
        self.mode = mode.lower()
        if self.mode not in ['max', 'mean', 'sum']:
            raise ValueError("Unsupported pooling mode. Supported modes are: 'max', 'mean', 'sum'.")

    def forward(self, x):
        if self.mode == 'max':
            return torch.max(x, dim=1)[0]
        elif self.mode == 'mean':
            return torch.mean(x, dim=1)
        elif self.mode == 'sum':
            return torch.sum(x, dim=1)
        else:
            raise RuntimeError("Invalid pooling mode.")

    def get_output_shape(self):
        return self.input_shape[1]




class SelectNode(nn.Module):
    def __init__(self, node, index_start, index_end=None):
        super().__init__()
        self.index_start = index_start
        self.index_end = index_end if index_end is not None else index_start + 1 
        shape = node.get_output_shape()
        assert isinstance(shape, tuple) and len(shape) == 2, "Expected the shape to be a 2D tuple"
        # Adjust the output shape since we are selecting along dim1
        self.output_shape = (self.index_end - self.index_start, shape[1])

    def forward(self, x):
        selected = x[:, self.index_start:self.index_end, :]  # This assumes x has shape (batch_size, dim1, dim2)
        return selected

    def get_output_shape(self):
        return self.output_shape