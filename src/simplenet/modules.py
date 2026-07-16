import numpy as np

from simplenet.tensor import Tensor

# refactor utility functions
def calculate_output_shape(height, width, kernel_size, stride, padding):
    kernel_height, kernel_width = kernel_size
    # output kernel dimensions, not using dilation for now (set to 1)
    out_height = (height + 2 * padding - kernel_height) // stride + 1
    out_width = (width + 2 * padding - kernel_width) // stride + 1
    return out_height, out_width

def im2col(x: Tensor, kernel_size, stride, padding, pad_value=0):
    # x: (batch, channels, height, width)
    # pad with 0 for convolution, -inf for max pooling, since 0 could be the max value.
    kernel_height, kernel_width = kernel_size
    batch_size, channels, height, width = x.data.shape
    out_height, out_width = calculate_output_shape(height, width, kernel_size, stride, padding)

    # pad the input
    x_padded = np.pad(x.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=pad_value)

    # im2col
    cols_data = np.zeros((batch_size, channels * kernel_height * kernel_width, out_height * out_width))
    for oy in range(out_height):
        for ox in range(out_width):
            patch = x_padded[:, :, oy*stride:oy*stride+kernel_height, ox*stride:ox*stride+kernel_width]
            cols_data[:, :, oy*out_width + ox] = patch.reshape(batch_size, -1)

    cols = Tensor(cols_data, (x,))
    def _backward():
        # col2im is exactly the adjoint (scatter-add) of im2col's gather,
        # so it's reused here as im2col's backward.
        x.grad += col2im(Tensor(cols.grad), x.data.shape, kernel_size, stride, padding).data
    cols._backward_fn = _backward
    return cols

def col2im(cols: Tensor, x_shape, kernel_size, stride, padding):
    # cols: (batch, channels * kernel_height * kernel_width, out_height * out_width)
    kernel_height, kernel_width = kernel_size
    batch_size, channels, height, width = x_shape
    out_height, out_width = calculate_output_shape(height, width, kernel_size, stride, padding)

    x_padded = np.zeros((batch_size, channels, height + 2 * padding, width + 2 * padding))
    for oy in range(out_height):
        for ox in range(out_width):
            patch = cols.data[:, :, oy*out_width + ox].reshape(batch_size, channels, kernel_height, kernel_width)
            x_padded[:, :, oy*stride:oy*stride+kernel_height, ox*stride:ox*stride+kernel_width] += patch
    if padding > 0:
        return Tensor(x_padded[:, :, padding:-padding, padding:-padding])
    return Tensor(x_padded)

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

    def forward(self, x):
        # x: (batch, channels, height, width)
        kernel_size = (self.weight.data.shape[2], self.weight.data.shape[3])
        cols = im2col(x, kernel_size, self.stride, self.padding)
        # reshape weight to (out_channels, in_channels * kernel_size * kernel_size)
        weight_reshaped = self.weight.reshape((self.weight.data.shape[0], -1))
        # (out_channels, 1) so it broadcasts against the channel axis of
        # (batch, out_channels, out_height * out_width), not the spatial axis
        bias_reshaped = self.bias.reshape((self.bias.data.shape[0], 1))
        out = weight_reshaped @ cols + bias_reshaped
        # reshape back to (batch, out_channels, out_height, out_width)
        batch_size, _, height, width = x.data.shape
        out_height, out_width = calculate_output_shape(height, width, kernel_size, self.stride, self.padding)
        return out.reshape((batch_size, -1, out_height, out_width))

    def parameters(self):
        return [self.weight, self.bias]
    
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        # x: (batch, channels, height, width)
        batch_size, channels, height, width = x.data.shape
        kernel_size = (self.kernel_size, self.kernel_size)
        out_height, out_width = calculate_output_shape(height, width, kernel_size, self.stride, self.padding)

        # im2col flattens each window to channels * kernel_h * kernel_w, which
        # mixes channels together. Needs to be split into (channels, window) to take
        # max within a window for each channel separately. So reshape to (batch, channels, window, out_height * out_width)
        cols = im2col(x, kernel_size, self.stride, self.padding, pad_value=-np.inf)
        cols = cols.reshape((batch_size, channels, self.kernel_size * self.kernel_size, out_height * out_width))
        # find the max along the columns axis, which is actually the window
        out = cols.max(axis=2)
        return out.reshape((batch_size, channels, out_height, out_width))

    def parameters(self):
        return []

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
