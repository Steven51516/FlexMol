import torch
import numpy as np

def get_collate(x, collate_func):
    collate_func.append(None)
    transposed_x = list(zip(*x))
    collated_data = [func(data) if func is not None else torch.tensor(np.array(data)) for func, data in zip(collate_func, transposed_x)]
    return collated_data


def tuple_collate(x):
    return [torch.stack([torch.tensor(item) for item in list_item]) for list_item in zip(*x)]


