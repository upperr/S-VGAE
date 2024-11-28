import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate
import torch_sparse
from torch_geometric.nn import GCNConv

class SigmoidStochastic(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return torch.bernoulli(torch.sigmoid(x))

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha
        return grad_x, None

class SigmoidSurrogate(surrogate.SurrogateFunctionBase):
    """
    Sigmoid surrogate gradient
    """
    def __init__(self, alpha = 1.0, spiking = True):
        '''
        alpha: parameter to control smoothness of gradient
        spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        '''
        super(SigmoidSurrogate, self).__init__(alpha, spiking)
    
    @staticmethod
    def spiking_function(x, alpha):
        return SigmoidStochastic.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()

class SpikingFC(nn.Module):
    """
    Spiking fully connected layer
    """
    def __init__(self, input_dim, output_dim, bias = False, dropout = 0., bn = True):
        super(SpikingFC, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.linear = nn.Linear(input_dim, output_dim, bias = bias)
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(input_dim) if bn else None

    def forward(self, inputs):
        
        if self.dropout:
            inputs = F.dropout(inputs, self.dropout, training = self.training)
        if self.bn is not None:
            inputs = self.bn(inputs)
        outputs = self.linear(inputs) # [N, C_l] -> [N, C_(l+1)]
        
        return outputs

class GraphConvolution(nn.Module):
    """
    Spiking graph convolution layer
    """
    def __init__(self, dropout = 0.):
        super(GraphConvolution, self).__init__()
        
        self.dropout = dropout

    def forward(self, inputs):
        x, adj = inputs
        if self.dropout:
            x = F.dropout(x, self.dropout, training = self.training)
        outputs = torch_sparse.matmul(adj, x)  # [N, C]
        return outputs  # [N, C]

class GCNLayer(nn.Module):
    """
    Graph convolution layer
    """
    def __init__(self, input_dim, output_dim, act = nn.ReLU(), normalize = False, bias = False, dropout = 0., bn = True):
        super(GCNLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.GraphConv = GCNConv(input_dim, output_dim, normalize = normalize, bias = bias)
        self.act = act
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(input_dim) if bn else None

    def forward(self, inputs):
        x, adj = inputs
        if self.dropout:
            x = F.dropout(x, self.dropout, training = self.training)
        if self.bn is not None:
            x = self.bn(x)
        outputs = self.GraphConv(x, adj)  # (N, K)
        outputs = self.act(outputs)  # (N, K)
        return outputs

class WeightedInnerProduct_Link(nn.Module):
    """
    Decoder model layer for link prediction
    """
    def __init__(self, input_dim):
        super(WeightedInnerProduct_Link, self).__init__()
        
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1, bias = False)

    def forward(self, inputs):
        x_s, x_t = inputs
        x = x_s * x_t
        x = self.linear(x)
        outputs = torch.sum(x, dim = 1) # [N, C] -> [N]
        return outputs

class WeightedInnerProduct_Graph(nn.Module):
    """
    Decoder model layer for graph generation
    """
    def __init__(self, input_dim):
        super(WeightedInnerProduct_Graph, self).__init__()
        
        self.input_dim = input_dim
        self.weights = nn.Parameter(torch.empty((input_dim, 1)))
        self.weights = nn.init.xavier_uniform_(self.weights) # initialize weights

    def forward(self, inputs):
        x_s, x_t = inputs
        outputs = torch.matmul(x_s, x_t.t() * self.weights) # [N, C] -> [N, N]
        return outputs

class NonFiringLIF(nn.Module):
    """
    Output the last time membrane potential of the LIF neuron with V_th=infty
    """
    def __init__(self, T, tau = 0.8) -> None:
        super().__init__()

        arr = torch.arange(T - 1, -1, -1) # T
        self.register_buffer("coef", torch.pow(tau, arr)[None, :]) # (1, T)

    def forward(self, x):
        outputs = torch.sum(x * self.coef, dim = -1) # [N, T] -> N or [N, N, T] -> [N, N]
        return outputs

class PotentialOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        outputs = torch.mean(x, dim = -1) # [N, T] -> N
        return outputs