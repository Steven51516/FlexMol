import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import *

__all__ = ['MLP']

@register_to_device(True)
class MLP(EncoderLayer):
    """
    MLP is a multi-layer perceptron model.
    """

    def __init__(self, input_dim=1024, output_dim=128, hidden_dims_lst=[1024, 256, 64], device='cpu'):
        """
        Initialize the MLP model.

        Parameters:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            hidden_dims_lst (list): List of hidden layer dimensions.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        super(MLP, self).__init__()
        self.device = device
        layer_size = len(hidden_dims_lst) + 1
        dims = [input_dim] + hidden_dims_lst + [output_dim]
        self.output_shape = output_dim
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def get_output_shape(self):
        return self.output_shape

    @staticmethod
    def default_config(task, method):
        config_map = {}
        if task == "drug":
            config_map = {
                "Morgan": {"input_dim": 1024},
                "PubChem": {"input_dim": 881},
                "Daylight": {"input_dim": 2048},
                "ChemBERTa": {"input_dim": 384},
                "ErG": {"input_dim": 315},
                "ESPF": {"input_dim": 2586}
            }
        elif task == "prot_seq":
            config_map = {
                "AAC": {"input_dim": 8420},
                "ESPF": {"input_dim": 4114},
                "Conjoint_triad": {"input_dim": 343},
                "PseudoAAC": {"input_dim": 30},
                "Quasi-seq": {"input_dim": 100},
                "ESM": {"input_dim": 1280},
                "ProtTrans-t5": {"input_dim": 1024},
                "ProtTrans-albert": {"input_dim": 4096},
                "ProtTrans-bert": {"input_dim": 1024},
                "AutoCorr": {"input_dim": 720},
                "CTD": {"input_dim": 147},
            }
        return config_map.get(method, {})

    def forward(self, v):
        v = v.to(self.device).float()
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v
