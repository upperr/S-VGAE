import torch

def GraphNormalization(adj):
    adj = adj.set_diag()
    deg = adj.sum(dim = 1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj

def GraphListPreprocess(data_list):
    data_list_new = []
    for idx, data in enumerate(data_list):
        # normalize adjacency matrix for GCN
        adj = data.adj_t.set_diag()
        deg = adj.sum(dim = 1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        data.adj_t = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        # label adjacency matrix
        data.adj_orig = adj
        # feature
        if data.x is None:
            #data.x = torch_sparse.SparseTensor.eye(data.num_nodes, feature_dim)
            data.x = torch.ones((data.num_nodes, 1))
        data_list_new.append(data)
    return data_list_new
