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
        return x @ self.weight.transpose() + self.bias
    
    def parameters(self):
        return [self.weight, self.bias]
    
class Sequential(Module):
    def __init__(self, *layers): 
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# activation function modules to make it easier to build a neural network
class ReLU(Module):
    def forward(self, x):
        return x.relu()
    
    def parameters(self):
        return []
    
class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

    def parameters(self):
        return []
    
class Tanh(Module):
    def forward(self, x):
        return x.tanh()

    def parameters(self):
        return []
    
class Softmax(Module): 
    def forward(self, x):
        return x.softmax()

    def parameters(self):
        return []
