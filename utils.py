import torch
import numpy as np

SMALL = 1e-16

def kl_bernoulli(p_prior, p_posterior):
    """
    prob_prior: hyperparameter for Bernoulli prior
    prob_posterior: posterior probability for Bernoulli samples [N, C]
    """
    output = p_posterior * torch.log(SMALL + p_posterior / p_prior) + (1 - p_posterior) * torch.log(SMALL + (1. - p_posterior) / (1. - p_prior))

    return torch.mean(torch.sum(output, dim = (1, 2)))

def count_propagation(model, inputs, adj):
    x = inputs.cpu()
    ops_mul = int(adj.nnz() * np.count_nonzero(x) / adj.size(0))
    ops_ac = ops_mul - x.size(0)
    return ops_mul, ops_ac

def count_transformation(model, inputs):
    x = inputs.cpu()
    ops_mul = np.count_nonzero(x) * model.output_dim
    ops_ac = ops_mul - x.size(0)
    if model.bias:
        ops_ac += x.size(0) * model.output_dim
    return ops_mul, ops_ac

def count_GCN(model, inputs, adj):
    x = inputs.cpu()
    ops_mul = np.count_nonzero(x) * model.output_dim
    ops_ac = np.count_nonzero(x) * model.output_dim + adj.nnz() * model.output_dim - 2 * x.size(0)
    if model.bias:
        ops_ac += x.size(0) * model.output_dim
    return ops_mul, ops_ac

def count_inner_product_link(model, inputs):
    # average FLOPs for link prediction
    x1, x2 = inputs[0].cpu(), inputs[1].cpu()
    ops_mul = np.count_nonzero(x1 * x2)
    ops_ac = ops_mul - x1.size(0)
    return ops_mul, ops_ac

def count_inner_product_graph(model, inputs):
    # average FLOPs for graph generation
    x1, x2 = inputs[0].cpu(), inputs[1].cpu()
    ops_mul = int(np.count_nonzero(x1) * np.count_nonzero(x2) / x1.size(0))
    ops_ac = ops_mul - x1.size(0)
    return ops_mul, ops_ac

def count_readout(model, inputs):
    x = inputs.cpu()
    ops_mul = x.size(0)
    ops_ac = x.size(0) * (x.size(1) - 1)
    return ops_mul, ops_ac

def count_encoding(inputs, adj, adj_2):
    ops_mul = int(adj.nnz() * adj.nnz() / adj.size(0) + adj_2.nnz() * inputs.size(1))
    ops_ac = ops_mul - 2 * inputs.size(0)
    return ops_mul, ops_ac