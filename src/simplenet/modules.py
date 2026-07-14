import numpy as np

from simplenet.tensor import Tensor

class Module:
    def parameters(self):
        return []
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights and biases with random values by pytorch framework defaults
        self.weight = Tensor(np.random.randn(out_features, in_features) * np.sqrt(1. / in_features))
        self.bias = Tensor(np.random.randn(out_features) * np.sqrt(1. / in_features))
    
    def forward(self, x):
        return x @ self.weight + self.bias
    
    def parameters(self):
        return [self.weight, self.bias]