from torch.utils.data import Dataset
import copy

class FMDataset(Dataset):
    def __init__(self, df, encoders):
        self.features = df.iloc[:, :-1].values
        self.labels = df.iloc[:, -1].values
        self.encoders = encoders

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row_features = copy.deepcopy(self.features[index])
        label = self.labels[index]
        for i, encoder in enumerate(self.encoders):
            if encoder.model_training_setup["loadtime_transform"]:
                row_features[i] = encoder.featurizer(row_features[i], mode="loadtime")
        return (*row_features, label)

