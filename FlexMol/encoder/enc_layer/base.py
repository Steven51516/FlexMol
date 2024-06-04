import torch
import torch.nn as nn
from abc import ABC, abstractmethod


__all__ = [
    'EncoderLayer',
    'register_to_device',
    'register_loadtime_transform',
    'register_collate_func',
]


class EncoderLayer(nn.Module, ABC):
    _training_setup = {
        "collate_func": None, 
        "to_device_in_model": False, 
        "loadtime_transform": False
    }

    def __init__(self):
        super(EncoderLayer, self).__init__()

    @abstractmethod
    def get_output_shape(self):
        """This should be overridden by all subclasses to return the output shape based on instance configurations."""
        pass

    @classmethod
    def training_setup(cls):
        """Return the training setup configuration specific to the class."""
        return cls._training_setup

    @staticmethod
    def default_config(task, method):
        """Return the default configuration settings for the encoder layer."""
        return {}


def create_training_setup_decorator(key):
    """Creates a decorator factory for setting a specified training setup key with a user-provided value at the class level."""
    def decorator_factory(value):
        def decorator(cls):
            if '_training_setup' not in cls.__dict__:
                cls._training_setup = cls._training_setup.copy()
            cls._training_setup[key] = value
            return cls
        return decorator
    return decorator_factory

register_to_device = create_training_setup_decorator('to_device_in_model')
register_loadtime_transform = create_training_setup_decorator('loadtime_transform')
register_collate_func = create_training_setup_decorator('collate_func')