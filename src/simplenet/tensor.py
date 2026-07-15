import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, children: tuple = ()):
        self.data = data
        self.grad = np.zeros_like(data)
        self._children = set(children)  # root node is the final output, so the children are the previous nodes
        self._backward_fn = None

    def _unbroadcast(self, grad, shape):
        # sum out any leading dims that were added by broadcasting
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # sum out any dims that were size-1 in the original but broadcast up
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=np.float64))
        out = Tensor(self.data + other.data, (self, other))
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward_fn = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=np.float64))
        out = Tensor(self.data * other.data, (self, other))
        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward_fn = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __exp__(self):
        out = Tensor(np.exp(self.data), (self,))
        def _backward():
            self.grad += out.grad * out.data
        out._backward_fn = _backward
        return out
    
    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,))
        def _backward():
            self.grad += out.grad * power * (self.data ** (power - 1))
        out._backward_fn = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __neg__(self):
        out = Tensor(-self.data, (self,))
        def _backward():
            self.grad += -out.grad
        out._backward_fn = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), (self,))
        def _backward():
            self.grad += out.grad * np.sign(self.data) # gradient of abs is 1, x > 0, -1, x < 0
        out._backward_fn = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))
        def _backward():
            self.grad += out.grad / self.data
        out._backward_fn = _backward
        return out

    # matrix operations
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        def _backward():
            # swapaxes(-1, -2) transposes only the matrix dims, not any
            # leading broadcast/batch dims (unlike .T, which reverses all
            # axes and is only correct for plain 2D matmul).
            other_t = np.swapaxes(other.data, -1, -2)
            self_t = np.swapaxes(self.data, -1, -2)
            self.grad += self._unbroadcast(out.grad @ other_t, self.data.shape)
            other.grad += self._unbroadcast(self_t @ out.grad, other.data.shape)
        out._backward_fn = _backward
        return out
    
    def reshape(self, new_shape):
        out = Tensor(self.data.reshape(new_shape), (self,))
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward_fn = _backward
        return out
    
    def transpose(self):
        out = Tensor(self.data.T, (self,))
        def _backward():
            self.grad += out.grad.T
        out._backward_fn = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad
        out._backward_fn = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        count = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,))
        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad / count
        out._backward_fn = _backward
        return out

    # activation functions
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))
        def _backward():
            self.grad += out.grad * (self.data > 0) # gradient of relu is 1 for positive inputs, 0 for negative inputs, standard is 0 at x = 0
        out._backward_fn = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,))
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data) # gradient of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        out._backward_fn = _backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,))
        def _backward():
            self.grad += out.grad * (1 - out.data ** 2) # gradient of tanh is 1 - tanh(x)^2
        out._backward_fn = _backward
        return out
    
    def softmax(self, axis=-1):
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out = Tensor(exp_data / np.sum(exp_data, axis=axis, keepdims=True), (self,))
        def _backward():
            s = out.data
            g = out.grad
            self.grad += s * (g - np.sum(g * s, axis=axis, keepdims=True)) # gradient of softmax is s * (g - sum(g * s))
        out._backward_fn = _backward
        return out
    
    # backward pass
    def backward(self):
        # topological sort of the computation graph
        topo_order = []
        visited = set()
        def dfs(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._children:
                    dfs(child)
                topo_order.append(tensor)
        dfs(self)

        # initialize the gradient of the output tensor to 1
        self.grad = np.ones_like(self.data)

        # traverse the graph in reverse topological order and call the backward function for each tensor
        for tensor in reversed(topo_order):
            if tensor._backward_fn is not None:
                tensor._backward_fn()

    def _get_shape(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            else:
                return (len(data),) + self._get_shape(data[0])
        else:
            return ()
        
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,))
        def _backward():
            grad = np.zeros_like(self.data)
            # add all indices in idx to the gradient of the original tensor
            np.add.at(grad, idx, out.grad)
            self.grad += grad
        out._backward_fn = _backward
        return out

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"