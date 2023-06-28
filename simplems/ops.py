"""Operatpr table."""
# Global operator table.
from typing import Optional

from . import init
from .autograd import (Tensor, TensorOp, TensorTuple,
                       TensorTupleOp, Value)
from .backend_selection import NDArray, array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)

        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)

class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)

class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # 避免标量情形
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        input, = node.inputs
        # nx^(n-1)
        return out_grad * (self.scalar * array_api.power(input, self.scalar - 1))

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""
    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs / rhs

def divide(a, b):
    return EWiseDiv()(a, b)

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # to keep dtype
        return (a / self.scalar)

    def gradient(self, out_grad, node):
        return out_grad / self.scalar

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        shape = list(range(len(a.shape)))
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        shape[x], shape[y] = shape[y], shape[x]
        return array_api.permute(a, tuple(shape))

    def gradient(self, out_grad, node):
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return transpose(out_grad, axes=(x, y))

def transpose(a, axes=None):
    return Transpose(axes)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        input, = node.inputs
        return reshape(out_grad, input.shape)

def reshape(a, shape):
    return Reshape(shape)(a)

##### batch size = 1特殊处理
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
    def compute(self, a):
        if a.shape == self.shape:
            return a
        else:
            return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input, = node.inputs
        if input.shape == self.shape:
            return out_grad
        # 找到广播的维度
        # input: scalar
        n1 = len(input.shape)
        n2 = len(self.shape)
        # 计算系数
        c = 1
        # 扩充为相同维度大小
        shape = [1] * (n2 - n1) + list(input.shape)
        axes = []
        for i in reversed(range(n2)):
            # 不相等或填充为0
            if shape[i] != self.shape[i] or i < n2 - n1:
                axes.append(i)
        return reshape(summation(out_grad, axes=tuple(axes)), input.shape)
        ## END YOUR SOLUTION

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        n = len(a.shape)
        axes = []
        # 处理多维度求和
        if not isinstance(self.axes, tuple):
            ori_axes = self.axes,
        else:
            ori_axes = self.axes
        for axis in ori_axes:
            # 处理负数情形
            if isinstance(axis, int):
                if axis < 0:
                    axes.append(axis + n)
                else:
                    axes.append(axis)
            else:
                axes.append(axis)
        # 降序排列
        axes = sorted(axes, reverse=True)
        for axis in axes:
            a = array_api.sum(a, axis)
        return a

    def gradient(self, out_grad, node):
        input, = node.inputs
        # 使坐标为正并且从小到大排列
        if self.axes == None:
            axes = input.shape
            grad_shape = []
        else:
            axes = self.axes
            grad_shape = list(out_grad.shape)

        n = len(input.shape)
        new_axes = []
        for x in axes:
            if x >= 0:
                new_axes.append(x)
            else:
                new_axes.append(x + n)
        new_axes = sorted(new_axes)
        # 恢复grad_shape, 使grad_shape的维度和input.shape的维度相同
        for axis in new_axes:
            grad_shape.insert(axis, 1)
        return broadcast_to(reshape(out_grad, grad_shape), input.shape)

def summation(a, axes=None):
    return Summation(axes)(a)

class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        # (A, a, b), (B, b, c)
        lhs, rhs = node.inputs
        # out_grad: (C, a, c)
        # (C, a, b)
        lhs_grad = matmul(out_grad, transpose(rhs, axes=(-1, -2)))
        # (C, b, c)
        rhs_grad = matmul(transpose(lhs, axes=(-1, -2)), out_grad)
        # 注意形状
        n1 = len(out_grad.shape)
        n2 = len(lhs.shape)
        n3 = len(rhs.shape)
        if n1 > n2:
            lhs_grad = summation(lhs_grad, axes=tuple(range(n1 - n2)))
        if n1 > n3:
            rhs_grad = summation(rhs_grad, axes=tuple(range(n1 - n3)))
        return lhs_grad, rhs_grad

def matmul(a, b):
    return MatMul()(a, b)

class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad

def negate(a):
    return Negate()(a)

class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        input, = node.inputs
        return out_grad / input

def log(a):
    return Log()(a)

class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        input, = node.inputs
        return out_grad * exp(input)

def exp(a):
    return Exp()(a)

class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        input, = node.inputs
        input_relu = relu(input).realize_cached_data()
        return out_grad * Tensor(input_relu > 0, device=out_grad.device, dtype=out_grad.dtype)

def relu(a):
    return ReLU()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        z = array_api.max(Z, axis=self.axes, keepdims=True)
        z_broadcast = array_api.broadcast_to(z, Z.shape)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - z_broadcast), axis=self.axes, keepdims=True)) + z

        new_shape = []
        if self.axes:
            l = len(Z.shape)
            for i, n in enumerate(Z.shape):
                if (i not in self.axes) and ((i - l) not in self.axes):
                    new_shape.append(n)
            log_sum_exp = log_sum_exp.reshape(new_shape)  # .astype(Z.dtype)
        return log_sum_exp

    def gradient(self, out_grad, node):
        input, = node.inputs
        data = input.realize_cached_data()
        z = array_api.max(data, axis=self.axes, keepdims=True)
        z = array_api.broadcast_to(z, input.shape)
        e = array_api.exp(data - z)
        e_sum = array_api.sum(e, axis=self.axes, keepdims=True)
        e_sum = array_api.broadcast_to(e_sum, input.shape)
        prob = e / e_sum
        new_shape = list(input.shape)
        # (a, b) -> (1, a, 1, b)
        if self.axes:
            for i in self.axes:
                new_shape[i] = 1
            grad = reshape(out_grad, new_shape)
        else:
            grad = out_grad

        return broadcast_to(grad, input.shape) * Tensor(prob, dtype=grad.dtype, device=grad.device)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        input, = node.inputs
        tmp = tanh(input)
        return out_grad * (1 - tmp * tmp)

def tanh(a):
    return Tanh()(a)

def getitem(x, axises):
    for axis in axises:
        x = make_tuple(x)[axis]
    return x

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        n = len(args)
        shape = list(args[0].shape)
        arg_shape = list(args[0].shape)
        shape.insert(self.axis, n)
        new_arr = array_api.empty(shape, dtype=args[0].dtype, device=args[0].device)
        # 计算index
        idxes = []
        m = len(arg_shape)
        for i in range(m):
            idxes.append(slice(0, arg_shape[i]))
        idxes.insert(self.axis, 0)
        # 新形状
        arg_shape.insert(self.axis, 1)

        # 赋值
        for i in range(n):
            idxes[self.axis] = i
            new_arr[tuple(idxes)] = array_api.reshape(args[i], arg_shape)
        return new_arr
    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)

def stack(args, axis):
    return Stack(axis)(make_tuple(*args))

class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # (5, 3, 5) -> [(5, 5), (5, 5), (5, 5)]
        n = A.shape[self.axis]
        arg_shape = list(A.shape)
        new_arr = []
        # 计算index
        idxes = []
        m = len(arg_shape)
        for i in range(m):
            idxes.append(slice(0, arg_shape[i]))
        # 新形状
        new_shape = list(A.shape)
        del new_shape[self.axis]

        # 赋值
        for i in range(n):
            idxes[self.axis] = i
            data = array_api.array(A[tuple(idxes)], dtype=A.dtype, device=A.device)
            data = array_api.reshape(data, new_shape)
            new_arr.append(data)

        return new_arr

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)

def flip(a, axes):
    return Flip(axes)(a)

class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        old_shape = list(a.shape)
        new_shape = []
        n = len(old_shape)
        index = []
        for i in range(n):
            if i not in self.axes:
                new_shape.append(old_shape[i])
                index.append(slice(new_shape[-1]))
            else:
                new_shape.append(old_shape[i] * (1 + self.dilation))
                index.append(slice(0, new_shape[-1], 1 + self.dilation))

        res = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        res[tuple(index)] = a

        return res

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        old_shape = list(a.shape)
        new_shape = []
        n = len(old_shape)
        index = []
        for i in range(n):
            if i not in self.axes:
                new_shape.append(old_shape[i])
                index.append(slice(new_shape[-1]))
            else:
                new_shape.append(old_shape[i] // (1 + self.dilation))
                index.append(slice(0, old_shape[i], 1 + self.dilation))

        res = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        res = a[tuple(index)]

        return res

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        axes = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        A_pad = array_api.pad(A, axes)
        N, H, W, C_in = A_pad.shape
        K, _, _, C_out = B.shape
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        out = array_api.full(
            shape=(N, H_out, W_out, C_out),
            fill_value=0,
            dtype=A.dtype,
            device=A.device
        )
        # for i in range(K):
        #     for j in range(K):
        #         out += Z[:,i:i+H-K+1,j:j+W-K+1,:] @ weight[i,j]
        batch_index = slice(N)
        feature_index1 = slice(C_in)
        feature_index2 = slice(C_out)
        n = N * H_out * W_out
        for i in range(K):
            for j in range(K):
                # 不要和dilate搞混
                i_start = i
                i_end = i_start + H_out * self.stride
                h_index = slice(i_start, i_end, self.stride)
                j_start = j
                j_end = j_start + W_out * self.stride
                w_index = slice(j_start, j_end, self.stride)
                A1 = A_pad[(batch_index, h_index, w_index, feature_index1)]
                A2 = array_api.reshape(A1, (n, C_in))
                B1 = B[slice(i, i + 1, 1), slice(j, j + 1, 1), feature_index1, feature_index2]
                B2 = array_api.reshape(B1, (C_in, C_out))
                C2 = array_api.matmul(A2, B2)
                C3 = array_api.reshape(C2, (N, H_out, W_out, C_out))
                out += C3

        return out

    def gradient(self, out_grad, node):
        # out_grad: bhwc2
        out_grad_dilate = dilate(out_grad, (1, 2), self.stride - 1)
        # A: bhwc1, B: kkc1c2
        A, B = node.inputs
        A = A.realize_cached_data()
        B = B.realize_cached_data()
        b = A.shape[0]
        h = A.shape[1]
        k = B.shape[0]
        # bhwc1 -> c1hwb
        A = array_api.permute(A, (3, 1, 2, 0))
        # kkc1c2 -> kkc1c2
        B = array_api.flip(B, (0, 1))
        # kkc1c2 -> kkc2c1
        B = array_api.permute(B, (0, 1, 3, 2))
        tmp = ((h + 2 * self.padding - k) // self.stride + 1) * self.stride
        # pad
        p_B = (h + k - tmp - 1) // 2
        p_A = (k + tmp - h - 1) // 2
        # bhwc2, kkc2c1 -> bhwc1
        grad_A = conv(out_grad_dilate, Tensor(B, dtype=out_grad.dtype, device=out_grad.device), stride=1, padding=p_B)
        # bhwc2 -> whbc2 -> hwbc2: out_grad_dilate.transpose((0, 2)).transpose((0, 1))
        # c1hwb, hwbc2 -> c1hwc2
        grad_B = conv(Tensor(A, dtype=out_grad.dtype, device=out_grad.device),
                      out_grad_dilate.transpose((0, 2)).transpose((0, 1)), stride=1, padding=p_A)
        # c1hwc2 -> whc1c2 -> hwc1c2
        grad_B = grad_B.transpose((0, 2)).transpose((0, 1))

        return grad_A, grad_B


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
