from .FM_base import *
from collections import OrderedDict
import torch.nn as nn
from .FM_config import *


class DrugEncoder(EncoderNode):
    """ Specific encoder for drug data. """
    def __init__(self, method, pickle_dir=None, use_hash = True, custom_method = None, **config):
        super().__init__(method, pickle_dir, use_hash, **config)
        self.input_type = FlexMol.DRUG
        self.model, self.featurizer, self.model_training_setup = init_method(self.method, config, "drug", custom_method)


class ProteinSEQEncoder(EncoderNode):
    """ Specific encoder for protein sequence data. """
    def __init__(self, method, pickle_dir=None, use_hash = False, custom_method = None, **config):
        super().__init__(method, pickle_dir, use_hash, **config)
        self.input_type = FlexMol.PROT_SEQ
        self.model, self.featurizer, self.model_training_setup = init_method(self.method, config, "prot_seq", custom_method)


class ProteinPDBEncoder(EncoderNode):
    """ Specific encoder for protein 3D structure data. """
    def __init__(self, method, pickle_dir=None, use_hash = False, custom_method = None, **config):
        super().__init__(method, pickle_dir, use_hash, **config)
        self.input_type = FlexMol.PROT_3D
        self.model, self.featurizer, self.model_training_setup = init_method(self.method, config, "prot_3d", custom_method)


class FlexMol:
    """
    BioEncoder class for initializing and managing different types of encoders
    (Drug, Protein Sequence, Protein PDB) and setting up interactions between them.

    Attributes:
        nodes (list): List of nodes representing individual encoders or interaction layers.
        encoders_dic (OrderedDict): Dictionary mapping input types to encoder nodes.
        encoder_factory (dict): Dictionary mapping input types to encoder classes for instantiation.

    Methods:
        init_drug_encoder(model, **config): Initializes and registers a drug encoder.
        init_prot_encoder(model, pdb=False, **config): Initializes and registers a protein encoder, optionally for PDB data.
        init_encoder(encoder_type, model, **config): Generic initializer for encoders, registers them based on type.
        set_interaction(nodes, method, **config): Creates and registers an interaction node among specified encoder nodes.
        cat(nodes, **config): Creates and registers a concatenation interaction among specified nodes.
        stack(nodes, **config): Creates and registers a stacking interaction among specified nodes.
        flatten(node, **config): Creates and registers a flatten operation for a specified node.
        pooling(node, **config): Creates and registers a pooling operation for a specified node.
        apply_mlp(node, **config): Creates and registers a Multi-Layer Perceptron (MLP) for a specified node.
        build_model(): Compiles and returns a nn.Module representing the entire model composed of registered nodes.
        get_encoders(): Returns a list of all registered encoders.
        get_model(): Retrieves the compiled nn.Module of the encoder system.
    """


    DRUG = "Drug"
    PROT_SEQ = "Protein"
    PROT_3D = "Protein_ID"

    def __init__(self):
        self.nodes = []
        self.encoders_dic = OrderedDict()
        self.encoder_factory = {
            FlexMol.DRUG: DrugEncoder,
            FlexMol.PROT_SEQ: ProteinSEQEncoder,
            FlexMol.PROT_3D: ProteinPDBEncoder
        }
        self.model = None
        self.custom_method = {
            "drug" : {}, 
            "prot_seq": {},
            "prot_3d":{}
        }

    def register_method(self, type, method_name, encoder_layer_class, featurizer_class):
        self.custom_method[type][method_name] = (encoder_layer_class, featurizer_class)

    def init_drug_encoder(self, model, pickle_dir = None, use_hash = False, **config):
        """Initializes a drug encoder and registers it under the drug category."""
        return self.init_encoder(FlexMol.DRUG, model, pickle_dir, use_hash, self.custom_method, **config)

    def init_prot_encoder(self, model, pdb=False, pickle_dir = None, use_hash = False, **config):
        """Initializes a protein encoder and registers it. Chooses between sequence or PDB based on the pdb flag."""        
        if pdb:
            return self.init_encoder(FlexMol.PROT_3D, model, pickle_dir, use_hash, self.custom_method, **config)
        return self.init_encoder(FlexMol.PROT_SEQ, model, pickle_dir, use_hash, self.custom_method, **config)

    def init_encoder(self, encoder_type, model, pickle_dir = None, use_hash = False, custom_method = None, **config):
        """Instantiates an encoder of the specified type."""
        encoder = self.encoder_factory[encoder_type](model, pickle_dir, use_hash, custom_method, **config)
        encoder_node = NodeWrapper(model=encoder.get_model(), root_idx=len(self.encoders_dic),
                              output_shape=encoder.get_output_shape())
        self.nodes.append(encoder_node)
        self.encoders_dic[encoder] = encoder_node
        return encoder

    def set_interaction(self, nodes, method, **config):
        """Sets up and registers a specified interaction method among provided nodes."""
        if not isinstance(nodes, list):
            nodes = [nodes]
        if(method == "cat"):
            return self.cat(nodes, **config)
        nodes = [self.encoders_dic[node] if isinstance(node, EncoderNode) else node for node in nodes]
        inter_model = InteractionNode(nodes, method, **config)
        inter_node = NodeWrapper(parents=nodes, model=inter_model, output_shape=inter_model.get_output_shape())
        self.nodes.append(inter_node)
        return inter_node

    def cat(self, nodes, **config):
        """Creates a concatenation node that combines outputs of the provided nodes and registers it."""
        return self._create_composite_node(ConcatNode, nodes, **config)

    def stack(self, nodes, **config):
        """Creates a stacking node that stacks the outputs of the provided nodes vertically and registers it."""
        return self._create_composite_node(StackNode, nodes, **config)

    def flatten(self, node, **config):
        """Creates a flattening operation for the specified node and registers it."""
        return self._create_single_node(FlattenNode, node, **config)

    def pooling(self, node, **config):
        """Creates a pooling operation for the specified node and registers it."""
        return self._create_single_node(PoolingNode, node, **config)

    def select(self, node, **config):
        """Creates a seleccting operation for the specified node and registers it."""
        return self._create_single_node(SelectNode, node, **config)

    def apply_mlp(self, node, **config):
        """Applies an MLP transformation to the output of the specified node and registers the MLP node."""
        return self._create_single_node(MLPNode, node, **config)


    def _create_composite_node(self, node_class, nodes, **config):
        """
        Creates and registers a composite node, which combines multiple nodes.
        
        Args:
            node_class (class): The class of the node to create, e.g., ConcatNode, StackNode.
            nodes (list): List of nodes to combine.
            **config: Additional configuration parameters for the node class.
            
        Returns:
            NodeWrapper: A wrapped node instance representing the composite node.
        """
        nodes = [self.encoders_dic[node] if isinstance(node, EncoderNode) else node for node in nodes]
        composite_node = node_class(nodes, **config)
        composite_node_wrapper = NodeWrapper(parents=nodes, model=composite_node, output_shape=composite_node.get_output_shape())
        self.nodes.append(composite_node_wrapper)
        return composite_node_wrapper
    

    def _create_single_node(self, node_class, node, **config):
        """
        Creates and registers a single operation node, which applies a transformation to one node's output.
        
        Args:
            node_class (class): The class of the node to create, e.g., FlattenNode, PoolingNode.
            node (Node or EncoderNode): The node to apply the transformation.
            **config: Additional configuration parameters for the node class.
            
        Returns:
            NodeWrapper: A wrapped node instance representing the operation node.
        """
        node = self.encoders_dic[node] if isinstance(node, EncoderNode) else node
        operation_node = node_class(node, **config)
        operation_node_wrapper = NodeWrapper(parents=[node], model=operation_node, output_shape=operation_node.get_output_shape())
        self.nodes.append(operation_node_wrapper)
        return operation_node_wrapper
    

    def build_model(self):
        self.model = BEModel(self.nodes)

    def get_encoders(self):
        return list(self.encoders_dic.keys())

    def get_model(self):
        return self.model

    def set_device(self, device):
        encoders = self.get_encoders()
        for encoder in encoders:
            if encoder.model.training_setup()["to_device_in_model"]:
                encoder.model.device = device

    def clear(self):
        """
        Clears the internal state of the FlexMol instance, resetting all configurations.
        """
        self.nodes = []
        self.encoders_dic = OrderedDict()

        self.model = None
        self.custom_method = {
            "drug": {},
            "prot_seq": {},
            "prot_3d": {}
        }






class BEModel(nn.Module):
    """
    A neural network model that dynamically constructs a computation graph based on provided nodes.
    
    The model supports complex interactions and workflows by structuring itself according to
    a layered topological sort of the provided nodes. This allows for flexible, modular construction
    of neural network architectures that may include diverse processing layers and interactions.

    Attributes:
        nodes (list): A list of nodes, where each node represents an encapsulated computation unit.
        layers (list): Processed nodes arranged in sequential layers according to dependencies.
        input_indices_sequence (list): Indices mapping inputs of each node to outputs of previous nodes.
    """
    def __init__(self, nodes):
        super(BEModel, self).__init__()
        self.nodes = nodes
        self.layers, self.input_indices_sequence = self.layered_topological_sort()

        for i, node in enumerate(self.nodes):
            model = node.get_model()
            if model is not None and isinstance(model, (nn.Module, EncoderLayer)):
                setattr(self, f"module_{i}", model)

    def layered_topological_sort(self):
        layers = []
        processed = set()
        input_indices_sequence = []

        current_layer = [node for node in self.nodes if node.is_root()]
        input_indices_sequence.append([[node.get_root_idx()] for node in current_layer])
        indices = {}
        processed.update(current_layer)
        current_index = len(input_indices_sequence[0])
        while current_layer:
            layers.append(current_layer)
            layer_input_indices = []
            next_layer = []

            for node in current_layer:
                indices[node] = current_index
                current_index += 1

            for node in self.nodes:
                if node not in processed and all(parent in processed for parent in node.get_parents()):
                    layer_input_indices.append([indices[parent] for parent in node.get_parents()])
                    next_layer.append(node)
            input_indices_sequence.append(layer_input_indices)
            processed.update(next_layer)
            current_layer = next_layer

        return layers, input_indices_sequence

    def forward(self, *x):
        current_outputs = list(x)
        for layer, layer_input_indices in zip(self.layers, self.input_indices_sequence):
            next_outputs = []
            for node, input_indices in zip(layer, layer_input_indices):
                inputs = [current_outputs[index] for index in input_indices]
                if len(inputs) == 1:
                    if isinstance(inputs[0], tuple):
                        next_outputs.append(node.get_model()(*inputs[0]))
                    else:
                        next_outputs.append(node.get_model()(inputs[0]))
                else:
                    next_outputs.append(node.get_model()(*inputs))
            current_outputs.extend(next_outputs)
        return current_outputs[-1]