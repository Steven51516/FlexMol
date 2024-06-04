import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class InteractionLayer(nn.Module, ABC):

    def __init__(self):
        super(InteractionLayer, self).__init__()

    @abstractmethod
    def get_output_shape(self):
        """This should be overridden by all subclasses to return the output shape based on instance configurations."""
        pass

    @staticmethod
    def default_config(method):
        """Return the default configuration settings for the encoder layer."""
        return {}

