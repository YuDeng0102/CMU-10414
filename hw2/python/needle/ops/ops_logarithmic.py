from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z=array_api.max(Z,axis=self.axes,keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z-max_z),axis=self.axes))+max_z.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z=node.inputs[0]
        if self.axes == None:
            self.axes=tuple([i for i in range(len(z.shape))])

        shape=[1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        grad=exp(z-node.reshape(shape).broadcast_to(z.shape))
        return grad*out_grad.reshape(shape).broadcast_to(z.shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

