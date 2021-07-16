#!/home/virtual_envs/gcn_libs/bin/python3
import os, sys
sys.path.append('../')
from sage_tabular_model import SAGE_tabular
import torch
import dgl
import multiprocessing
import argparse
import configparser
import pickle as pk
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from dataset_preprocessor import mynormalize
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-s',  '--dataset', type=str, required=True)
    parser.add_argument('-d', '--dest', type=str, required=True)
    parser.add_argument('-t', '--targ_col', type=str, required=True)
    parser.add_argument('-f', '--feat_cols', type=str, required=True)
    parser.add_argument('--cat_cols', type=str, required=True)
    parser.add_argument('-a', '--adj', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('--config_option', type=str, required=False)
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--test_size', type=float, required=False, default=0.15)
    parser.add_argument('--nepochs', type=int, default=35)
    parser.add_argument('--early_stopping_rounds', type=int, default=10, dest='st_rounds')
    parser.add_argument('--task', type=str, default='reg') 

    args = parser.parse_args()
#   writer
    writer = SummaryWriter(f'{args.dest}/logs')

#   adjacency
    with open(args.adj, 'rb') as fp:
        adj = pk.load(fp)
        adj = mynormalize(sp.eye(adj.shape[0]) + adj)

#   dataset
    dataset = pd.read_pickle(args.dataset)
    with open(args.feat_cols, 'rb') as fp:
        feat_cols = pk.load(fp)
    with open(args.cat_cols, 'rb') as fp:
        cat_cols = pk.load(fp)

    if args.targ_col in dataset.columns:
        targ_col = [args.targ_col]
    else:
        with open(args.targ_col, 'rb') as fp:
            targ_col = pk.load(fp)

        
#    import pdb; pdb.set_trace()
    dataset = dataset.sort_values(['num'])
    labels = dataset[targ_col]
    labeled_ids = torch.arange(len(dataset))[~labels.isna().values.ravel()]
    train_ids, test_ids = train_test_split(labeled_ids, test_size = args.test_size, random_state=args.seed)
    labels = torch.FloatTensor(labels.values)


#   graph
    g = dgl.graph(adj.nonzero(), num_nodes = adj.shape[0])
    g.ndata['feats'] = torch.FloatTensor(dataset[feat_cols].values)
    for c in cat_cols:
        g.ndata[c] = torch.LongTensor(dataset[c].values)

#   config
    config = configparser.ConfigParser()
    config.read(args.config)
    tab_net_params = eval(config[args.config_option]['tab_net'])
    tab_net_params['dims'] = [len(feat_cols)+len(cat_cols)*tab_net_params['cat_dims']]+tab_net_params['dims']
    tab_net_params['cats'] = dataset[cat_cols].nunique().to_dict() 
    graph_net_params = eval(config[args.config_option]['graph_net'])
    graph_net_params['n_classes'] = len(targ_col)
    sample_nn_size = config.getint(args.config_option, 'sample_nn_size')
    batch_size = config.getint(args.config_option, 'batch_size')

#   model
    model = SAGE_tabular(tab_net_params, graph_net_params)
    pretrain_path = os.path.expanduser(config[args.config_option].get('pretrained', None))
    pretrained_keys = os.path.expanduser(config[args.config_option].get('pretrained_keys', None))
    if pretrain_path is not None:
        with open(pretrained_keys, 'rb') as fp:
            keys = pk.load(fp)
        state_dict = torch.load(pretrain_path)
        state_dict = {i:state_dict[i] for i in keys}
        model.load_state_dict(state_dict, strict=False)

#   dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler([sample_nn_size]*graph_net_params['n_layers'])
    dataloader = dgl.dataloading.NodeDataLoader(g, train_ids, sampler, batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0)
#            num_workers=multiprocessing.cpu_count())
    
    
    sampler_val = dgl.dataloading.MultiLayerFullNeighborSampler(graph_net_params['n_layers'])
    dataloader_val = dgl.dataloading.NodeDataLoader(g, test_ids, sampler_val, batch_size=10,
            shuffle=True,
            drop_last=True,
            num_workers=0)

#   task
    optimizer = torch.optim.Adam(model.parameters())
    if args.task == 'clf':
        loss = torch.nn.MultiLabelSoftMarginLoss()
    else:
        loss = torch.nn.MSELoss()
    ebar = tqdm(range(args.nepochs))
    min_loss = float('inf')
    step = 0

    for epoch in ebar:
        model.train()
        for input_nodes, output_nodes, blocks in tqdm(dataloader, leave=False, total=len(train_ids)//batch_size):
            optimizer.zero_grad()
            feats = blocks[0].srcdata['feats']
            cat = {c:blocks[0].srcdata[c] for c in cat_cols}

            out = model(blocks, feats, cat)
            loss_ = loss(out, labels[output_nodes])
            loss_.backward()
            optimizer.step()
            train_loss = loss_.item()
            writer.add_scalar('train_loss', train_loss, global_step=step)
            step += 1

        counter = 0 
        val_loss = 0
        pred, true = [], []
        model.eval()
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in tqdm(dataloader_val, leave=False, total=len(test_ids)//10):
                feats = blocks[0].srcdata['feats']
                cat = {c:blocks[0].srcdata[c] for c in cat_cols}
                out = model(blocks, feats, cat)
                loss_ = loss(out, labels[output_nodes])
                val_loss += loss_.item()
                pred += [out]
                true += [labels[output_nodes]]
                counter += 1

        val_loss /= counter
        true = torch.cat(true)
        pred = torch.cat(pred)
        if args.task == 'reg':
            score = 1-mean_absolute_percentage_error(true.numpy(), pred.numpy())
        else:
            score = roc_auc_score(true.numpy(), pred.numpy())
        writer.add_scalar('valid_loss', val_loss, global_step=epoch)
        writer.add_scalar('valid_score', score, global_step=epoch)
        if args.task == 'reg':
            ebar.set_postfix_str(s=f'score={1-score}, loss={val_loss}', refresh=True)
        else:
            ebar.set_postfix_str(s=f'score={score}, loss={val_loss}', refresh=True)
    
        if val_loss < min_loss:
            min_loss = val_loss
            rounds_to_stop = args.st_rounds
            torch.save(model.state_dict(), f'{args.dest}/weights')
        
        else:
            rounds_to_stop -= 1
            if rounds_to_stop < 0:
                break
