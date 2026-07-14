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
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # same initialization as pytorch from kaiming uniform or somethin lol
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(1. / (in_channels * kernel_size * kernel_size)))
        self.bias = Tensor(np.random.randn(out_channels) * np.sqrt(1. / (in_channels * kernel_size * kernel_size)))
        self.stride, self.padding = stride, padding

    def _out_hw(self, height, width):
        kernel_height, kernel_width = self.weight.data.shape[2], self.weight.data.shape[3]
        # output kernel dimensions, not using dilation for now (set to 1)
        out_height = (height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_width) // self.stride + 1
        return out_height, out_width

    def im2col(self, x):
        # x: (batch, channels, height, width)
        batch_size, channels, height, width = x.data.shape
        kernel_height, kernel_width = self.weight.data.shape[2], self.weight.data.shape[3]
        out_height, out_width = self._out_hw(height, width)

        # pad the input
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # im2col
        cols_data = np.zeros((batch_size, channels * kernel_height * kernel_width, out_height * out_width))
        for oy in range(out_height):
            for ox in range(out_width):
                patch = x_padded[:, :, oy*self.stride:oy*self.stride+kernel_height, ox*self.stride:ox*self.stride+kernel_width]
                cols_data[:, :, oy*out_width + ox] = patch.reshape(batch_size, -1)

        cols = Tensor(cols_data, (x,))
        def _backward():
            # col2im is exactly the adjoint (scatter-add) of im2col's gather,
            # so it's reused here as im2col's backward.
            x.grad += self.col2im(Tensor(cols.grad), x.data.shape).data
        cols._backward_fn = _backward
        return cols

    def col2im(self, cols, x_shape):
        # cols: (batch, channels * kernel_height * kernel_width, out_height * out_width)
        batch_size, channels, height, width = x_shape
        kernel_height, kernel_width = self.weight.data.shape[2], self.weight.data.shape[3]
        out_height, out_width = self._out_hw(height, width)

        x_padded = np.zeros((batch_size, channels, height + 2 * self.padding, width + 2 * self.padding))
        for oy in range(out_height):
            for ox in range(out_width):
                patch = cols.data[:, :, oy*out_width + ox].reshape(batch_size, channels, kernel_height, kernel_width)
                x_padded[:, :, oy*self.stride:oy*self.stride+kernel_height, ox*self.stride:ox*self.stride+kernel_width] += patch
        if self.padding > 0:
            return Tensor(x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding])
        return Tensor(x_padded)

    def forward(self, x):
        # x: (batch, channels, height, width)
        cols = self.im2col(x)
        # reshape weight to (out_channels, in_channels * kernel_size * kernel_size)
        weight_reshaped = self.weight.reshape((self.weight.data.shape[0], -1))
        # (out_channels, 1) so it broadcasts against the channel axis of
        # (batch, out_channels, out_height * out_width), not the spatial axis
        bias_reshaped = self.bias.reshape((self.bias.data.shape[0], 1))
        out = weight_reshaped @ cols + bias_reshaped
        # reshape back to (batch, out_channels, out_height, out_width)
        batch_size, _, height, width = x.data.shape
        out_height, out_width = self._out_hw(height, width)
        return out.reshape((batch_size, -1, out_height, out_width))

    def parameters(self):
        return [self.weight, self.bias]

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
