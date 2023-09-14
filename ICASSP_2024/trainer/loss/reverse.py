import torch
import torch.nn as nn
import torch.nn.functional as F

from   torch.autograd import Variable
import torch.nn.init as init

## GradReverse ##
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_output = grad_output * ctx.constant
        # print('梯度反转后的梯度：',grad_output)
        return -grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)