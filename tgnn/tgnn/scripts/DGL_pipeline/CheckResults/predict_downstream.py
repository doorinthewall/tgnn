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
from collections import defaultdict


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
    parser.add_argument('--pretrained', type=str, required=True)

    args = parser.parse_args()

#   adjacency
    with open(args.adj, 'rb') as fp:
        adj = pk.load(fp)
        adj = mynormalize(sp.eye(adj.shape[0]) + adj)

#   dataset
    dataset = pd.read_pickle(args.dataset)    
    if args.targ_col in dataset.columns:
        targ_col = [args.targ_col]
    else:
        with open(args.targ_col, 'rb') as fp:
            targ_col = pk.load(fp)
    with open(args.feat_cols, 'rb') as fp:
        feat_cols = pk.load(fp)
    with open(args.cat_cols, 'rb') as fp:
        cat_cols = pk.load(fp)

    dataset = dataset.sort_values(['num'])
    labels = dataset[targ_col]
    labeled_ids = torch.arange(len(dataset))[~labels.isna().values.ravel()]

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

#   model
    model = SAGE_tabular(tab_net_params, graph_net_params)
    pretrain_path = args.pretrained
    state_dict = torch.load(pretrain_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

#   dataloader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(graph_net_params['n_layers'])
    dataloader = dgl.dataloading.NodeDataLoader(g, labeled_ids, sampler, batch_size=10,
            shuffle=True,
            drop_last=True,
            num_workers=0)

#   inference
    result = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm(dataloader, leave=False, total=len(labeled_ids)//10):
            feats = blocks[0].srcdata['feats']
            cat = {c:blocks[0].srcdata[c] for c in cat_cols}
            out = model(blocks, feats, cat)
            result['pred'] += out.detach().flatten().tolist()
            result['num'] += list(dataset.iloc[output_nodes.flatten().tolist()]['num'].values)
    result = pd.DataFrame(result)
    result.to_pickle(args.dest)
