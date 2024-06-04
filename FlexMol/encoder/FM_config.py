from .enc_layer import *
from .featurizer import *
import inspect
from FlexMol.encoder.Inter_layer import *
from torch.nn.utils.weight_norm import weight_norm


def process_configs(model_class, featurizer_class, config):
    """
    Splits the configuration parameters between the model and featurizer.
    
    Args:
    - model_class: The class of the model.
    - featurizer_class: The class of the featurizer.
    - config: The combined configuration dictionary.

    Returns:
    - Tuple of two dictionaries: (model_params, featurizer_params).
    """
    model_args = inspect.signature(model_class.__init__).parameters
    featurizer_args = inspect.signature(featurizer_class.__init__).parameters
    
    model_params = {k: v for k, v in config.items() if k in model_args}
    featurizer_params = {k: v for k, v in config.items() if k in featurizer_args}
    
    return model_params, featurizer_params


def init_method(method, user_config, type, custom_method = None):
    methods = {
        "drug": {
            "CNN": (CNN, DrugOneHotFeaturizer),
            "GCN": (DGL_GCN, DrugCanonicalFeaturizer),
            "GAT": (DGL_GAT, DrugCanonicalFeaturizer),
            "NeuralFP": (DGL_NeuralFP, DrugCanonicalFeaturizer),
            "Transformer": (Transformer, DrugESPFTransFeaturizer),
            "Morgan": (MLP, MorganFeaturizer),
            "Daylight": (MLP, DaylightFeaturizer),
            "ErG": (MLP, ErGFeaturizer),
            "PubChem": (MLP, PubChemFeaturizer), 
            "ChemBERTa": (MLP, ChemBERTaFeaturizer),
            "ESPF": (MLP, DrugESPFFeaturizer),
            "MPNN": (DGL_MPNN, Drug3dFeaturizer),
            "SchNet": (DGL_SchNet, Drug3dFeaturizer),
            "MGCN": (DGL_MGCN, Drug3dFeaturizer),
            "GIN": (DGL_GIN, DrugCanonicalFeaturizer),
            "AttentiveFP": (DGL_AttentiveFP, DrugAttentiveFPFeaturizer)
        },
        "prot_seq": {
            "CNN": (CNN, ProteinOneHotFeaturizer),
            "Transformer": (Transformer, ProteinESPFTransFeaturizer),
            "AAC": (MLP, ProteinAACFeaturizer),
            "ESPF": (MLP, ProteinESPFFeaturizer),
            "PseudoAAC":(MLP, ProteinPAACFeaturizer),
            "Quasi-seq":(MLP, ProteinQuasiFeaturizer),
            "Conjoint_triad":(MLP, ProteinCTFeaturizer),
            "CTD:" : (MLP, ProteinCTDFeaturizer),
            "AutoCorr" : (MLP, ProteinAutoCorrFeaturizer)
 
        },
        "prot_3d": {
            "GCN": (DGL_GCN, ProteinGraphFeaturizer),
            "GAT": (DGL_GAT, ProteinGraphFeaturizer),
            "NeuralFP": (DGL_NeuralFP, ProteinGraphFeaturizer),
            "GCN_ESM": (DGL_GCN, ProteinGraphESMFeaturizer),
            "GAT_ESM": (DGL_GAT, ProteinGraphESMFeaturizer),
            "PocketDC":(TAG, PocketDeepChemFeaturizer),
            "GVP":(MQAModel, ProteinGVPFeaturizer),
            "GearNet" :(GeometryAwareRelationalGraphNeuralNetwork, ProteinTorchDrug),
        }
    }

    if custom_method:
        model_class, featurizer_class = custom_method[type].get(method, (None, None))
        if model_class is None or featurizer_class is None:
            model_class, featurizer_class = methods[type].get(method, (None, None))
    else:
        model_class, featurizer_class = methods[type].get(method, (None, None))

    if not model_class:  
        raise ValueError(f"Unsupported method: {method}")

    model_config, featurizer_config = process_configs(model_class, featurizer_class, user_config)

    model_config = {**model_class.default_config(type, method), **model_config}

    return model_class(**model_config), featurizer_class(**featurizer_config), model_class.training_setup()




def init_inter_layer(method, parent_output_shapes, **config):
    methods = {
            "bilinear_attention": BANLayer,
            "self_attention": MultiHeadAttention,
            "cross_attention": BidirectionalCrossAttention,
            "highway" : Highway,
            "gated_fusion":GatedFusionLayer,
            "bilinear_fusion": BilinearFusion
    }

    inter_class = methods.get(method)
    if inter_class is None:
        raise ValueError(f"Unsupported method: {method}")
    method_config = {**inter_class.default_config(method), **config}

    if method == "bilinear_attention":
        if len(parent_output_shapes) != 2:
            raise ValueError("Bilinear interaction requires exactly two parent nodes.")
        inter_layer = weight_norm(inter_class(parent_output_shapes[0][1], parent_output_shapes[1][1], **method_config), name='h_mat', dim=None)

    elif method == "self_attention":
        if len(parent_output_shapes) != 1:
            raise ValueError("Self interaction requires exactly 1 parent node.")
        inter_layer = inter_class(parent_output_shapes[0][1], **method_config)

    elif method == "cross_attention":
        if len(parent_output_shapes) != 2:
            raise ValueError("Cross interaction requires exactly two parent nodes.")
        inter_layer = inter_class(parent_output_shapes[0][1], parent_output_shapes[1][1], **method_config)

    elif method == "highway":
        if len(parent_output_shapes) != 1:
            raise ValueError("Highway requires exactly 1 parent node.")
        inter_layer = inter_class(parent_output_shapes[0][1], **method_config)

    elif method == "gated_fusion":
        if len(parent_output_shapes) != 2:
            raise ValueError("Gated Fusion requires exactly two parent nodes.")
        inter_layer = inter_class(parent_output_shapes[0][1], parent_output_shapes[1][1], **method_config)

    elif method == "bilinear_fusion":
        if len(parent_output_shapes) != 2:
            raise ValueError("Bilinear Fusion requires exactly two parent nodes.")
        inter_layer = inter_class(parent_output_shapes[0][1], parent_output_shapes[1][1], **method_config)

    else:
        raise ValueError(f"Unknown method: {method}")


    output_shape = inter_layer.get_output_shape()
    if output_shape == "SAME_TO_PARENT":
        output_shape = parent_output_shapes[0]

    return inter_layer, output_shape

