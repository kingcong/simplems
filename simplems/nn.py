"""The module.
"""
import math
from typing import List
from collections import OrderedDict
from simplems.autograd import Tensor
from simplems import ops
import simplems.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module(object):
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # def __repr__(self):
    #     return ""

class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.use_bias = bias
        w = init.kaiming_uniform(in_features, out_features)
        self.weight = Parameter(w, device=device, dtype=dtype)
        if self.use_bias:
            b = ops.reshape(init.kaiming_uniform(out_features, 1), (1, out_features))
            self.bias = Parameter(b, device=device, dtype=dtype)
        

    def forward(self, X: Tensor) -> Tensor:
        
        # Y = XA + b, input shape X: (N, H_in), output shape Y: (N, H_out)
        # weight shape: (H_in, H_out)
        Y = ops.matmul(X, self.weight)
        if self.use_bias:
            bias = ops.broadcast_to(self.bias, Y.shape)
            Y += bias

        return Y
        


class Flatten(Module):
    def forward(self, X):
        
        l = len(X.shape)
        for i in range(l - 2):
            X_shape = X.shape
            d2 = X_shape[-2]
            d1 = X_shape[-1]
            new_shape = X_shape[:-2] + (d1 * d2,)
            X = ops.reshape(X, new_shape)

        return X
        


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
        


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)
        


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.exp(-ops.log(ops.add_scalar(ops.exp(-x), 1.0)))
        


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for moudle in self.modules:
            x = moudle(x)
        return x
        


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # output shape: （n, m）
        n = 1.0 * logits.shape[0]
        m = logits.shape[-1]
        y_one_hot = init.one_hot(m, y, device=logits.device, dtype=logits.dtype)
        z_y = ops.summation(ops.multiply(logits, y_one_hot), axes=(1,))

        return ops.divide_scalar(ops.summation(ops.logsumexp(logits, axes=(-1,)) - z_y), float(n))
        

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(self.dim),
                                device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_size = x.shape[0]
            mean = x.sum(axes=0) / batch_size
            x_mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ((x - x_mean) ** 2).sum(axes=0) / batch_size
            x_var = var.reshape((1, self.dim)).broadcast_to(x.shape)
            self.running_mean = ((1 - self.momentum) *
                                 self.running_mean + self.momentum * mean).data
            self.running_var = ((1 - self.momentum) *
                                self.running_var + self.momentum * var).data
        else:
            x_mean = self.running_mean.reshape(
                (1, self.dim)).broadcast_to(x.shape)
            x_var = self.running_var.reshape(
                (1, self.dim)).broadcast_to(x.shape)
        norm = (x - x_mean) / ((x_var + self.eps) ** 0.5)
        return self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * norm + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # dim默认是最后一维的dimension
        w = init.ones(dim)
        self.weight = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(dim)
        self.bias = Parameter(b, device=device, dtype=dtype)
        

    def forward(self, x: Tensor) -> Tensor:
        l = len(x.shape)
        # 倒数第二维
        n = x.shape[-2]
        # 最后一维，和self.dim相同
        d = x.shape[-1]
        # 1, ..., 1, d
        broadcast_shape = (1, d)
        for i in range(l - 2):
            broadcast_shape = (1, ) + broadcast_shape
        # ..., n, 1
        stat_shape = x.shape[:-2] + (n, 1)
        # mean
        # ..., d -> ...,
        x_mean = ops.summation(x, axes=-1) / d
        # ..., -> ..., d
        x_mean = ops.broadcast_to(ops.reshape(x_mean, stat_shape), x.shape)
        x_zero = x - x_mean
        # var
        # ..., d -> ...,
        x_var = ops.summation(ops.multiply(x_zero, x_zero), axes=-1) / d
        # ..., -> ..., d
        x_var = ops.broadcast_to(ops.reshape(x_var, stat_shape), x.shape)
        x_stan_var = ops.power_scalar(x_var + self.eps, 0.5)
        # normalize
        x_normalize = x_zero / x_stan_var
        # result
        weight = ops.broadcast_to(ops.reshape(self.weight, broadcast_shape), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, broadcast_shape), x.shape)
        res = x_normalize * weight + bias

        return res
        


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # prob = Tensor(np.random.binomial(n=1, p=1-self.p, size=x.shape))
            prob = init.randb(*x.shape, p=1-self.p,
                              device=x.device, dtype=x.dtype)
            return x * prob / (1-self.p)
        else:
            return x
        


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
        

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        
        self.padding = self.kernel_size // 2
        # kernel
        fan_in = self.in_channels * self.kernel_size ** 2
        fan_out = self.out_channels * self.kernel_size ** 2
        shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        weight = init.kaiming_uniform(fan_in, fan_out, shape=shape)
        self.weight = Parameter(weight, device=device, dtype=dtype)
        # bias
        self.use_bias = bias
        if self.use_bias:
            k = 1.0 / (self.in_channels * self.kernel_size ** 2) ** 0.5
            b = ops.reshape(init.uniform(self.out_channels, 1, k), (self.out_channels,))
            self.bias = Parameter(b, device=device, dtype=dtype)
        

    def forward(self, x: Tensor) -> Tensor:
        # NCHW -> NHCW -> NHWC
        x = ops.transpose(x, (2, 1))
        x = ops.transpose(x, (3, 2))
        # NHWC
        output = ops.conv(x, self.weight, self.stride, self.padding)
        # 1C -> 111C -> NHWC
        if self.use_bias:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            bias = ops.broadcast_to(bias, output.shape)
            output += bias
        output = ops.transpose(output, (3, 2))
        output = ops.transpose(output, (2, 1))

        return output
        


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        
        self.use_bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = math.sqrt((1 / hidden_size))
        w1 = init.uniform(input_size, hidden_size, k)
        self.W_ih = Parameter(w1, device=device, dtype=dtype)
        w2 = init.uniform(hidden_size, hidden_size, k)
        self.W_hh = Parameter(w2, device=device, dtype=dtype)
        if self.use_bias:
            b1 = init.uniform(1, hidden_size, k)
            self.bias_ih = Parameter(b1, device=device, dtype=dtype)
            b2 = init.uniform(1, hidden_size, k)
            self.bias_hh = Parameter(b2, device=device, dtype=dtype)
        if nonlinearity == "tanh":
            self.f = Tanh()
        else:
            self.f = ReLU()
        

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        bs = X.shape[0]
        if h == None:
            h = Tensor(init.zeros(bs, self.hidden_size), device=X.device, dtype=X.dtype)
        tmp = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        if self.use_bias:
            tmp += ops.broadcast_to(ops.reshape(self.bias_ih, (1, self.hidden_size)), tmp.shape) + \
                   ops.broadcast_to(ops.reshape(self.bias_hh, (1, self.hidden_size)), tmp.shape)
        h = self.f(tmp)
        return h
        


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                d = input_size
            else:
                d = hidden_size
            rnn_cells.append(RNNCell(d, hidden_size, bias, nonlinearity, device, dtype))
        self.rnn_cells = rnn_cells
        

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        n = X.shape[0]  # sequence length
        bs = X.shape[1]  # batch size

        # (l, bs, h_out)
        if h0 == None:
            h0 = Tensor(init.zeros(self.num_layers, bs, self.hidden_size), device=X.device, dtype=X.dtype)
        # (l, bs, h_out) -> [(b, e), ... , (b, e)]
        h = ops.split(h0, axis=0)
        # (n, bs, h_out) -> [(b, e), ... , (b, e)]
        X_split = ops.split(X, axis=0)
        h_out = []
        for j in range(self.num_layers):
            h_state = h[j]
            X_state = []
            for i in range(n):
                h_state = self.rnn_cells[j](X_split[i], h_state)
                X_state.append(h_state)
            h_out.append(h_state)
            X_split = X_state

        return ops.stack(X_split, 0), ops.stack(h_out, 0)
        

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.exp(-ops.log(ops.add_scalar(ops.exp(-x), 1.0)))

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        
        self.use_bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = math.sqrt(1 / hidden_size)
        w1 = init.uniform(input_size, 4*hidden_size, k)
        self.W_ih = Parameter(w1, device=device, dtype=dtype)
        w2 = init.uniform(hidden_size, 4*hidden_size, k)
        self.W_hh = Parameter(w2, device=device, dtype=dtype)
        if self.use_bias:
            b1 = init.uniform(1, 4*hidden_size, k)
            self.bias_ih = Parameter(b1, device=device, dtype=dtype)
            b2 = init.uniform(1, 4*hidden_size, k)
            self.bias_hh = Parameter(b2, device=device, dtype=dtype)
        self.sigma = Sigmoid()
        self.f = Tanh()
        


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs = X.shape[0]
        if h == None:
            h0 = Tensor(init.zeros(bs, self.hidden_size), device=X.device, dtype=X.dtype)
            c0 = Tensor(init.zeros(bs, self.hidden_size), device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        # bs, hidden_size * 4
        tmp = ops.matmul(X, self.W_ih) + ops.matmul(h0, self.W_hh)
        if self.use_bias:
            tmp += ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4*self.hidden_size)), tmp.shape) + \
                   ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4*self.hidden_size)), tmp.shape)
        tmp = ops.reshape(tmp, (bs, 4, self.hidden_size))
        i, f, g, o = ops.split(tmp, 1)
        i = self.sigma(i)
        f = self.sigma(f)
        g = self.f(g)
        o = self.sigma(o)

        c = f * c0 + i * g
        h = o * self.f(c)

        return h, c
        


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        lstm_cells = []
        for i in range(num_layers):
            if i == 0:
                d = input_size
            else:
                d = hidden_size
            lstm_cells.append(LSTMCell(d, hidden_size, bias, device, dtype))
        self.lstm_cells = lstm_cells
        

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        n = X.shape[0]
        bs = X.shape[1]
        # (l, b, e)
        if h == None:
            h0 = Tensor(init.zeros(self.num_layers, bs, self.hidden_size), device=X.device, dtype=X.dtype)
            c0 = Tensor(init.zeros(self.num_layers, bs, self.hidden_size), device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        # (l, b, e) -> [(b, e), ... ,(b, e)]
        h = ops.split(h0, axis=0)
        c = ops.split(c0, axis=0)
        # (n, b, e) -> [(b, e), ... ,(b, e)]
        X_split = ops.split(X, axis=0)

        h_out = []
        c_out = []
        for j in range(self.num_layers):
            h_state = h[j]
            c_state = c[j]
            X_state = []
            for i in range(n):
                h_state, c_state = self.lstm_cells[j](X_split[i], (h_state, c_state))
                X_state.append(h_state)
            h_out.append(h_state)
            c_out.append(c_state)
            X_split = X_state

        return ops.stack(X_split, 0), (ops.stack(h_out, 0), ops.stack(c_out, 0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = init.randn(num_embeddings, embedding_dim)
        self.weight = Parameter(weight, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        # l, b, m
        x_one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        n, b, m = x_one_hot.shape
        # l, b, m -> l * b, m
        x_one_hot = ops.reshape(x_one_hot, (n * b, m))
        # l * b, d
        output = ops.matmul(x_one_hot, self.weight)
        # l, b, d
        output = ops.reshape(output, (n, b, self.embedding_dim))

        return output
        
