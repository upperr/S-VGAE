import os
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, TUDataset, ZINC
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

def load_data(dataset, seed = None, percent = 100):
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        path = os.path.join('../dataset', dataset)
        dataset = Planetoid(path, dataset)
        data = dataset[0]
        split_edge = do_edge_split(dataset, seed = seed)
        train_pos_edges, train_neg_edges = get_pos_neg_edges(split = 'train',
                                                             split_edge = split_edge,
                                                             num_nodes = dataset[0].num_nodes,
                                                             percent = percent)
        val_pos_edges, val_neg_edges = get_pos_neg_edges(split = 'valid',
                                                         split_edge = split_edge,
                                                         num_nodes = dataset[0].num_nodes,
                                                         percent = percent)
        test_pos_edges, test_neg_edges = get_pos_neg_edges(split = 'test',
                                                           split_edge = split_edge,
                                                           num_nodes = dataset[0].num_nodes,
                                                           percent = percent)
        # edge splits for link prediction
        data.edge_index = train_pos_edges
        # adjacency matrix
        data = ToSparseTensor(remove_edge_index = False)(data)
        data.neg_edge_index = train_neg_edges
        data.val_edge_index = val_pos_edges
        data.val_neg_edge_index = val_neg_edges
        data.test_edge_index = test_pos_edges
        data.test_neg_edge_index = test_neg_edges
        
    elif dataset in ['ogbl-collab', 'ogbl-ppa']:
        dataset = PygLinkPropPredDataset(dataset, root = '../dataset')
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        train_pos_edges, train_neg_edges = get_pos_neg_edges(split = 'train',
                                                             split_edge = split_edge,
                                                             num_nodes = dataset[0].num_nodes,
                                                             percent = percent)
        val_pos_edges, val_neg_edges = get_pos_neg_edges(split = 'valid',
                                                         split_edge = split_edge,
                                                         num_nodes = dataset[0].num_nodes,
                                                         percent = percent)
        test_pos_edges, test_neg_edges = get_pos_neg_edges(split = 'test',
                                                           split_edge = split_edge, 
                                                           num_nodes = dataset[0].num_nodes,
                                                           percent = percent)
        # adjacency matrix
        data = ToSparseTensor(attr = None)(data)
        # edge splits for link prediction
        data.edge_index = train_pos_edges
        data.neg_edge_index = train_neg_edges
        data.val_edge_index = val_pos_edges
        data.val_neg_edge_index = val_neg_edges
        data.test_edge_index = test_pos_edges
        data.test_neg_edge_index = test_neg_edges
        # node features
        data.x = data.x.type(torch.float32)
    else:
        raise Exception('Dataset name must be one of Cora, CiteSeer, PubMed, ogbl-collab, ogbl-ppa.')
    return data

def do_edge_split(dataset, val_ratio = 0.05, test_ratio = 0.1, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
    data = dataset[0]

    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(edge_index,
                                                  num_nodes = data.num_nodes,
                                                  num_neg_samples = data.train_pos_edge_index.size(1))

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge

def get_pos_neg_edges(split, split_edge, num_nodes, percent = 100):
    """
    Parameters
    ----------
    split: split name. train, valid or test
    split_edge: dictionary of splitted edges
    num_nodes: number of nodes
    percent: default 100.

    Returns
    -------
    pos_edge: tensor with shape [2, num_edges]
    neg_edge: tensor with shape [2, num_edges]
    """
    pos_edge = split_edge[split]['edge'].t()
    if split == 'train':
        new_edge_index, _ = add_self_loops(split_edge[split]['edge'].t())
        neg_edge = negative_sampling(new_edge_index,
                                     num_nodes = num_nodes,
                                     num_neg_samples = pos_edge.size(1))
    else:
        neg_edge = split_edge[split]['edge_neg'].t()
    # subsample for pos_edge
    num_pos = pos_edge.size(1)
    perm = np.random.permutation(num_pos)
    perm = perm[:int(percent / 100 * num_pos)]
    pos_edge = pos_edge[:, perm]
    # subsample for neg_edge
    num_neg = neg_edge.size(1)
    perm = np.random.permutation(num_neg)
    perm = perm[:int(percent / 100 * num_pos)] # subsample negative edges as the number of postive edges
    neg_edge = neg_edge[:, perm]

    return pos_edge, neg_edge

def load_graph_list(dataset, seed = None, val_ratio = 0.2, test_ratio = 0.3):
    if seed is not None:
        torch.manual_seed(seed)
    if dataset in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'COLLAB']:
        dataset = TUDataset(root = '../dataset', name = dataset)
        # data split
        split_graph = do_graph_split(dataset)
    elif dataset == 'ZINC':
        split_graph = {}
        split_graph['train'] = ZINC(root = os.path.join('../dataset', dataset), split = 'train')
        split_graph['valid'] = ZINC(root = os.path.join('../dataset', dataset), split = 'val')
        split_graph['test'] = ZINC(root = os.path.join('../dataset', dataset), split = 'test')
    elif dataset in ['ogbg-molhiv', 'ogbg-ppa']:
        dataset = PygGraphPropPredDataset(dataset, root = '../dataset') 
        split_idx = dataset.get_idx_split()
        data = {'train': [], 'valid': [], 'test': []}
        for i in range(len(split_idx['train'])):
            data['train'].append(ToSparseTensor(remove_edge_index = False)(dataset[i]))
            data['train'][-1].num_nodes = data['train'][-1].adj_t.size(0)
        for i in range(len(split_idx['valid'])):
            data['valid'].append(ToSparseTensor(remove_edge_index = False)(dataset[i]))
            data['valid'][-1].num_nodes = data['valid'][-1].adj_t.size(0)
        for i in range(len(split_idx['test'])):
            data['test'].append(ToSparseTensor(remove_edge_index = False)(dataset[i]))
            data['test'][-1].num_nodes = data['test'][-1].adj_t.size(0)
        return data
    else:
        raise Exception('Unrecognized dataset name.')
    data = {'train': [], 'valid': [], 'test': []}
    for i in range(len(split_graph['train'])):
        data['train'].append(ToSparseTensor(remove_edge_index = False)(split_graph['train'][i]))
        data['train'][-1].num_nodes = data['train'][-1].adj_t.size(0)
    for i in range(len(split_graph['valid'])):
        data['valid'].append(ToSparseTensor(remove_edge_index = False)(split_graph['valid'][i]))
        data['valid'][-1].num_nodes = data['valid'][-1].adj_t.size(0)
    for i in range(len(split_graph['test'])):
        data['test'].append(ToSparseTensor(remove_edge_index = False)(split_graph['test'][i]))
        data['test'][-1].num_nodes = data['test'][-1].adj_t.size(0)
    return data

def do_graph_split(dataset, val_ratio = 0.2, test_ratio = 0.3):        
    num_graphs = len(dataset)
    val_num_graphs = int(val_ratio * num_graphs)
    test_num_graphs = int(test_ratio * num_graphs)
    dataset = dataset.shuffle()
    # mask test and validation graphs
    split_graph = {}
    split_graph['valid'] = dataset[: val_num_graphs]
    split_graph['test'] = dataset[val_num_graphs: val_num_graphs + test_num_graphs]
    split_graph['train'] = dataset[val_num_graphs + test_num_graphs: ]
    return split_graph