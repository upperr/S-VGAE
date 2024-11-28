import os
import time
import random
import numpy as np
from tqdm import tqdm
import logging
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
# experimental options
parser.add_argument('--dataset', type = str, default = 'Cora', help = "Dataset name: Cora, CiteSeer, PubMed, ogbl-collab, ogbl-ppa")
parser.add_argument('--model', type = str, default = 'SVGAE', help = "Model name: SGAE, SVGAE")
parser.add_argument('--epochs', type = int, default = 1000, help = 'Max number of epochs to run. Training may stop early after convergence.')
parser.add_argument('--early_stopping', type = int, default = 100, help = "Number of epochs to run after last best validation.")
parser.add_argument('--batch_size', type = int, default = 512, help = 'Batch size.')
parser.add_argument('--num_workers', type = int, default = 0, help = "Number of workers.")
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate.')
parser.add_argument('--flops', action = 'store_true', help = "Whether to calculate FLOPs.")
parser.add_argument('--gpu', type = str, default = '0')
parser.add_argument('--seed', type = int, default = None, help = 'Random seed.')
# model options
parser.add_argument('--T', type = int, default = 10, help = 'Number of time steps.')
parser.add_argument('--encoder_layer', type = str, default = '64_64', help = 'Layer sizes for the encoder.')
parser.add_argument('--decoder_layer', type = int, default = 64, help = 'Layer size for the decoder.')
parser.add_argument('--dropout', type = float, default = 0.2, help = 'Dropout rate (1 - keep probability).')
parser.add_argument('--p_prior', type = float, default = 0.1, help = 'Probability for Bernoulli prior.')
parser.add_argument('--tau', type = float, default = 1.2, help = 'Membrane time constant of LIF neurons.')
parser.add_argument('--threshold', type = float, default = 0.2, help = 'Threshold voltage of LIF neurons.')
parser.add_argument('--reset', type = float, default = None, help = 'Reset voltage of LIF neurons.')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    exit(0)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from spikingjelly.clock_driven import encoding, functional

from load_data import load_data
from model import SGAE, SVGAE
from preprocess import GraphNormalization

class GraphDataset(Dataset):
    def __init__(self, data, split):
        if split == 'train':
            self.edge_index = data.edge_index
            self.neg_edge_index = data.neg_edge_index
        elif split == 'valid':
            self.edge_index = data.val_edge_index
            self.neg_edge_index = data.val_neg_edge_index
        elif split == 'test':
            self.edge_index = data.test_edge_index
            self.neg_edge_index = data.test_neg_edge_index

    def __getitem__(self, index):
        return self.edge_index.T[index], self.neg_edge_index.T[index]

    def __len__(self):
        return self.edge_index.size(1)

def train(model, edge_index, neg_edge_index, inputs, optimizer, epoch):
    
    total_loss = 0.
    total_kl = 0.
    num_batch = 0
    total_preds = []
    total_labels = []

    #train_loader = DataLoader(data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    train_loader = DataLoader(torch.arange(edge_index.size(1)), batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    model = model.train()
    
    #for pos_edges, neg_edges in tqdm(train_loader):
    for i in tqdm(train_loader):
        optimizer.zero_grad()
        pos_edges = edge_index.T[i]
        neg_edges = neg_edge_index.T[i]
        nodes_source = torch.cat([pos_edges[:, 0], neg_edges[:, 0]]).to(device)
        nodes_target = torch.cat([pos_edges[:, 1], neg_edges[:, 1]]).to(device)
        labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))]).to(device)
        
        preds = model(inputs, adj_t, nodes_source, nodes_target)
        loss, kl = model.loss_function(labels, kl_weight = epoch / args.epochs, p_prior = args.p_prior)
        loss.backward()
        optimizer.step()
        # reset, because snn has memory
        functional.reset_net(model)
        
        total_loss += loss.item()
        total_kl += kl.item()
        num_batch += 1
        total_preds.append(preds.detach().cpu())
        total_labels.append(labels.cpu())

    auc = roc_auc_score(torch.cat(total_labels), torch.cat(total_preds))
    ap = average_precision_score(torch.cat(total_labels), torch.cat(total_preds))
    
    return total_loss / num_batch, total_kl / num_batch, auc, ap

def test(model, edge_index, neg_edge_index, inputs, flops = False):

    total_preds = []
    total_labels = []
    total_muls = 0
    total_acs = 0
    num_batch = 0
    pos_test_loader = DataLoader(torch.arange(edge_index.size(1)), batch_size = 2 * args.batch_size, shuffle = False, num_workers = args.num_workers)
    neg_test_loader = DataLoader(torch.arange(neg_edge_index.size(1)), batch_size = 2 * args.batch_size, shuffle = False, num_workers = args.num_workers)
    model = model.eval()
    
    with torch.no_grad():
        for i in tqdm(pos_test_loader):
            pos_edges = edge_index.T[i]
            nodes_source = pos_edges[:, 0].to(device)
            nodes_target = pos_edges[:, 1].to(device)
            labels = torch.ones(pos_edges.size(0)).to(device)
            preds = model(inputs, adj_t, nodes_source, nodes_target, flops)
            # reset, because snn has memory
            functional.reset_net(model)
            
            total_preds.append(preds.cpu())
            total_labels.append(labels.cpu())
            # calculate FLOPs
            total_muls += model.muls
            total_acs += model.acs
            num_batch += 1
        for i in tqdm(neg_test_loader):
            neg_edges = neg_edge_index.T[i]
            nodes_source = neg_edges[:, 0].to(device)
            nodes_target = neg_edges[:, 1].to(device)
            labels = torch.zeros(neg_edges.size(0)).to(device)
            preds = model(inputs, adj_t, nodes_source, nodes_target, flops)
            # reset, because snn has memory
            functional.reset_net(model)
            
            total_preds.append(preds.cpu())
            total_labels.append(labels.cpu())
            # calculate FLOPs
            total_muls += model.muls
            total_acs += model.acs
            num_batch += 1
    
    auc = roc_auc_score(torch.cat(total_labels), torch.cat(total_preds))
    ap = average_precision_score(torch.cat(total_labels), torch.cat(total_preds))

    return auc, ap, int(total_muls / num_batch), int(total_acs / num_batch)

def log(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(filename = save_path, encoding = 'utf-8', mode = 'a')
    formator = logging.Formatter(fmt = "%(asctime)s %(filename)s %(levelname)s %(message)s",
                                         datefmt="%Y/%m/%d %X")
    sh.setFormatter(formator)
    fh.setFormatter(formator)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

if __name__ == '__main__':
    
    # Check whether a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    save_path = 'checkpoint/'
    os.makedirs(save_path, exist_ok = True)
    logger = log(save_path + f'{args.model}_{args.dataset}.log')
    logger.info(args)
    # load data
    logger.info("Loading dataset...")
    data = load_data(args.dataset)
    logger.info("Finished!")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    # transform adjacency matrix to normalized Laplacian
    adj_t = GraphNormalization(data.adj_t)
    inputs = data.x
    muls_enc = acs_enc = 0
    adj_t = adj_t.to(device, non_blocking = True)
    # node features as inputs
    inputs = inputs.to(device, non_blocking = True)
    
    # create model
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'ogbl-ppa']:
        SpikeEncoding = lambda x: x
    else:
        # spike inputs encoding
        SpikeEncoding = encoding.PoissonEncoder()
    if args.model == 'SVGAE':
        model = SVGAE(encoder_dim = [int(x) for x in args.encoder_layer.split('_')],
                  decoder_dim = args.decoder_layer,
                  n_features = data.x.shape[1],
                  inputs_encoding = SpikeEncoding,
                  task = 'link_prediction',
                  T = args.T,
                  dropout = args.dropout,
                  tau = args.tau,
                  threshold = args.threshold,
                  reset = args.reset).to(device)
    elif args.model == 'SGAE':
        model = SGAE(encoder_dim = [int(x) for x in args.encoder_layer.split('_')],
                  decoder_dim = args.decoder_layer,
                  n_features = data.x.shape[1],
                  inputs_encoding = SpikeEncoding,
                  task = 'link_prediction',
                  T = args.T,
                  dropout = args.dropout,
                  tau = args.tau,
                  threshold = args.threshold,
                  reset = args.reset).to(device)
        raise Exception('Unrecognized model.')
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, betas = (0.9, 0.999), weight_decay = 0.001)
    
    best_validation = 0
    last_best_epoch = 0
    for epoch in range(args.epochs):
        time_start = time.time()
        # training
        train_loss, train_kl, train_auc, train_ap = train(model, data.edge_index, data.neg_edge_index, inputs, optimizer, epoch)
        val_auc, val_ap, _, _ = test(model, data.val_edge_index, data.val_neg_edge_index, inputs)
        
        t = time.time() - time_start
        logger.info(f"Epoch: {epoch+1:04d} train_loss={train_loss:.3f} train_kl={train_kl:.3f} train_auc={train_auc:.3f} val_auc={val_auc:.3f} val_ap={val_ap:.3f} time={t:.2f}")

        if val_auc > best_validation:
            # save model
            torch.save(model.state_dict(), f'{save_path}{args.model}_{args.dataset}.pth')
            logger.info("Model saved!")
            last_best_epoch = 0
            best_validation = val_auc
        
        if last_best_epoch > args.early_stopping:
            break
        else:
            last_best_epoch += 1
    
    logger.info("Optimization Finished!")
    # testing
    model.load_state_dict(torch.load(f'{save_path}{args.model}_{args.dataset}.pth'))
    model.eval()
    test_auc, test_ap, muls, acs = test(model, data.test_edge_index, data.test_neg_edge_index, inputs, flops = args.flops)
    
    logger.info('Valid AUC score: ' + str(best_validation))
    logger.info('Test AUC score: ' + str(test_auc))
    logger.info('Test AP score: ' + str(test_ap))
    if args.flops:
        logger.info('Multiplications per edge: ' + str(muls + muls_enc))
        logger.info('Accumulations per edge: ' + str(acs + acs_enc))