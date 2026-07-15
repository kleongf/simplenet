import numpy as np

from simplenet.tensor import Tensor

class Dataset:
    def __len__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError

class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def train_test_split(dataset: Dataset, test_size=0.2, random_state=None):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(dataset))
    split_idx = int(len(dataset) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)

class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.data = x
        self.labels = y
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return Tensor(x), Tensor(y)

class TensorDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor, transform=None):
        self.data = x.data
        self.labels = y.data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return Tensor(x), Tensor(y)

