import numpy as np

from simplenet.tensor import Tensor

class Module:
    def parameters(self):
        return []
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

# hidden layers
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
    
class Flatten(Module):
    def forward(self, x):
        return x.reshape((x.data.shape[0], -1))

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True

    def forward(self, x):
        if self.training:
            mask = np.random.rand(*x.data.shape) > self.p
            mask = mask / (1 - self.p)  # scale the mask to normalize the output
            return x * Tensor(mask)
        else:
            return x

class BatchNorm1D(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Tensor(np.ones(num_features))
        self.beta = Tensor(np.zeros(num_features))
        # standard normal distribution with mean 0 and variance 1
        self.running_mean = Tensor(np.zeros(num_features))
        self.running_var = Tensor(np.ones(num_features))
        self.training = True

    def forward(self, x):
        if self.training:
            # Compute batch mean and variance by collapsing the batch dimension (axis=0), removes the first dimension
            batch_mean = x.mean(axis=0)
            batch_var = ((x - batch_mean) ** 2).mean(axis=0)
            # exponential moving average formula: (1-alpha) * old + x * new.
            # running stats are buffers, not trainable, so update .data directly instead of Tensor arithmetic
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * batch_mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * batch_var.data
            # formula for z-score normalization: (x - mean) / sqrt(var + eps), eps to prevent division by zero
            x_normalized = (x - batch_mean) / (batch_var + self.eps) ** 0.5
        else:
            x_normalized = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5

        out = self.gamma * x_normalized + self.beta
        return out
    
    def parameters(self):
        return [self.gamma, self.beta]

# activation function modules to make it easier to build a neural network
class ReLU(Module):
    def forward(self, x):
        return x.relu()
    
class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()
    
class Tanh(Module):
    def forward(self, x):
        return x.tanh()
    
class Softmax(Module): 
    def forward(self, x):
        return x.softmax()
