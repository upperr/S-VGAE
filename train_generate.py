import argparse
import time
import os
import random
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

# Settings
parser = argparse.ArgumentParser()
# experimental options
parser.add_argument('--dataset', type = str, default = 'MUTAG', help = "Dataset name: MUTAG, PTC_MR, ZINC")
parser.add_argument('--model', type = str, default = 'SVGAE', help = "Model name: SGAE, SVGAE")
parser.add_argument('--epochs', type = int, default = 100, help = 'Max number of epochs to run. Training may stop early after convergence.')
parser.add_argument('--early_stopping', type = int, default = 10, help = "Number of epochs to run after last best validation.")
parser.add_argument('--batch_size', type = int, default = 1, help = 'Batch size.')
parser.add_argument('--num_workers', type = int, default = 0, help = "Number of workers.")
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate.')
parser.add_argument('--flops', action = 'store_true', help = "Whether to calculate FLOPs.")
parser.add_argument('--gpu', type = str, default = '0')
parser.add_argument('--seed', type = int, default = None, help = 'Random seed.')
parser.add_argument('--seed_data', type = int, default = 1234, help = 'Random seed for dataset.')
# model options
parser.add_argument('--feature_dim', type = int, default = None, help = 'Dimension of input features.')
parser.add_argument('--T', type = int, default = 10, help = 'Number of time steps.')
parser.add_argument('--encoder_layer', type = str, default = '64_64', help = 'Layer sizes for the encoder.')
parser.add_argument('--decoder_layer', type = int, default = 64, help = 'Layer size for the decoder.')
parser.add_argument('--dropout', type = float, default = 0.2, help = 'Dropout rate (1 - keep probability).')
parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization. (default: False)')
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

from load_data import load_graph_list
from model import SGAE, SVGAE
from preprocess import GraphListPreprocess

class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def train(model, data, optimizer, epoch):
    
    model.train()
    train_loader = DataLoader(data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    nll = 0.
    num_batch = 0    
    for batch_data in tqdm(train_loader):
        adj_block = batch_data.adj_t.to(device)
        inputs = batch_data.x.float().to(device)
        label = batch_data.adj_orig.to_dense().to(device)
        pred = model(inputs, adj_block) # z(N, K, T)
        loss_curr, nll_curr = model.loss_function(label, kl_weight = epoch / args.epochs)
        optimizer.zero_grad()
        loss_curr.backward()
        optimizer.step()
        # reset, because snn has memory
        functional.reset_net(model)
        
        nll += nll_curr.item()
        num_batch += 1
    nll /= num_batch
    
    return nll

def test(model, data, flops = False):

    test_loader = DataLoader(data, batch_size = 1, num_workers = args.num_workers)
    model = model.eval()
    
    with torch.no_grad():
        nll = 0.
        auc = 0.
        ap = 0.
        muls = 0
        acs = 0
        num_batch = 0
        for batch_data in tqdm(test_loader):
            adj_block = batch_data.adj_t.to(device)
            inputs = batch_data.x.float().to(device)
            label = batch_data.adj_orig.to_dense().to(device)
            pred = model(inputs, adj_block, flops = flops) # z(N, K, T)
            _, nll_curr = model.loss_function(label, kl_weight = epoch / args.epochs)
            # reset, because snn has memory
            functional.reset_net(model)
            # calculate FLOPs
            muls += model.muls
            acs += model.acs
            
            label = label.cpu() - torch.diag_embed(label.cpu().diag())
            pred = pred.cpu() - torch.diag_embed(pred.cpu().diag())
            auc_curr = roc_auc_score(label.flatten(), pred.flatten())
            ap_curr = average_precision_score(label.flatten(), pred.flatten())
            nll += nll_curr.item()
            auc += auc_curr
            ap += ap_curr
            num_batch += 1
        nll /= num_batch
        auc /= num_batch
        ap /= num_batch
        muls /= num_batch
        acs /= num_batch

    return nll, auc, ap, int(muls), int(acs)

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
    data = load_graph_list(args.dataset, seed = args.seed_data)
    logger.info("Finished!")
    # set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    train_data = GraphListPreprocess(data['train'])
    valid_data = GraphListPreprocess(data['valid'])
    test_data = GraphListPreprocess(data['test'])
    muls_enc = acs_enc = 0
    feature_dim = train_data[0].x.shape[1]
    # construct Dataloader datasets
    train_data = GraphDataset(train_data)
    val_data = GraphDataset(valid_data)
    test_data = GraphDataset(test_data)
    
    # create model
    if args.dataset in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'COLLAB', 'ZINC']:
        SpikeEncoding = lambda x: x
    else:
        # spike inputs encoding
        SpikeEncoding = encoding.PoissonEncoder()
    if args.model == 'SGAE':
        model = SGAE(encoder_dim = [int(x) for x in args.encoder_layer.split('_')],
                  decoder_dim = args.decoder_layer,
                  n_features = feature_dim,
                  inputs_encoding = SpikeEncoding,
                  task = 'graph_generation',
                  T = args.T,
                  dropout = args.dropout,
                  tau = args.tau,
                  threshold = args.threshold,
                  reset = args.reset,
                  bn = args.bn).to(device)
    elif args.model == 'SVGAE':
        model = SVGAE(encoder_dim = [int(x) for x in args.encoder_layer.split('_')],
                  decoder_dim = args.decoder_layer,
                  n_features = feature_dim,
                  inputs_encoding = SpikeEncoding,
                  task = 'graph_generation',
                  T = args.T,
                  dropout = args.dropout,
                  tau = args.tau,
                  threshold = args.threshold,
                  reset = args.reset,
                  bn = args.bn).to(device)
    else:
        raise Exception('Unrecognized model.')
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, betas = (0.9, 0.999), weight_decay = 0.001)
    
    # training
    best_validation = 0
    last_best_epoch = 0
    for epoch in range(args.epochs):
        time_start = time.time()
        train_nll = train(model, train_data, optimizer, epoch)
        val_nll, val_auc, val_ap, _, _ = test(model, val_data)
        
        t = time.time() - time_start
        logger.info(f"Epoch: {epoch+1:04d} train_nll={train_nll:.3f} val_nll={val_nll:.3f} val_auc={val_auc:.3f} val_ap={val_ap:.3f} time={t:.2f}")
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
    # testing
    model.load_state_dict(torch.load(f'{save_path}{args.model}_{args.dataset}.pth'))
    model.eval()
    test_nll, test_auc, test_ap, muls, acs = test(model, test_data, args.flops)
    
    logger.info("Optimization Finished!")
    logger.info(f'Test NLL: {test_nll:.4f}')
    logger.info(f'Test AUC score: {test_auc:.4f}')
    logger.info(f'Test AP score: {test_ap:.4f}')
    if args.flops:
        logger.info('Multiplications per graph: ' + str(muls + int(muls_enc)))
        logger.info('Accumulations per graph: ' + str(acs + int(acs_enc)))