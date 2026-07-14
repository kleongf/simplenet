import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

