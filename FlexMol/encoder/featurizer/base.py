import numpy as np


class Featurizer:
    def __init__(self, data_dir = None):
        self.transform_modes = {
            'default': self.transform
        }
        self.data_dir = data_dir

    def __call__(self, x, mode="default", batch = False):
        """Apply the transformation to x using the specified mode."""
        transform_func = self.transform_modes.get(mode, self.transform_modes['default'])
        if batch:
            return self.transform_batch(x, mode)
        else:
            return transform_func(x)

    def transform(self, x):
        """Main transformation method."""
        raise NotImplementedError("Transform method needs to be implemented by the child class!")


    def transform_batch(self, x, mode):
        """Default Batch transformation method."""
        transform_func = self.transform_modes.get(mode, self.transform_modes['default'])
        if isinstance(x[0], np.ndarray):
            items_as_tuples = [tuple(item) for item in x]
            unique_tuples = list(set(items_as_tuples))
            unique_transformed = [transform_func(tuple_item) for tuple_item in unique_tuples]
            mapping = dict(zip(unique_tuples, unique_transformed))
            return [mapping[tuple(item)] for item in x]
        else:
            unique_values = np.unique(x)
            unique_transformed = [transform_func(item) for item in unique_values]
            mapping = dict(zip(unique_values, unique_transformed))
            return [mapping[item] for item in x]
