#!/home/Container/envs/bgnn_env/bin/python3
import os, sys
sys.path.append('DGL_pipeline')
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
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from collections import defaultdict
from data_utils import networkx_to_torch, normalize_features, replace_na, count_number_params
import json
import networkx as nx


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-s',  '--dataset', type=str, required=True)
    parser.add_argument('-d', '--dest', type=str, required=True)
    parser.add_argument('-t', '--target', type=str, required=True)
    parser.add_argument('--cat_cols', type=str, required=False)
    parser.add_argument('-g', '--graph', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('--config_option', type=str, required=False)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--masks', type=str, required=True)

    args = parser.parse_args()

#   dataset
    dataset = pd.read_csv(args.dataset)
    if args.cat_cols is not None:
        with open(args.cat_cols, 'rb') as fp:
            cat_cols = pk.load(fp)
    else:
        cat_cols = []

    with open(args.masks, 'rb') as fp:
        masks = json.load(fp)
        train_mask, val_mask, test_mask = masks[args.seed]['train'], masks[args.seed]['val'], masks[args.seed]['test']

    dataset = normalize_features(dataset, train_mask, val_mask, test_mask)
    dataset = replace_na(dataset, train_mask)

    labels = pd.read_csv(args.target)

#   graph
    networkx_graph = nx.read_graphml(args.graph)
    networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
    g = networkx_to_torch(networkx_graph)
    g.ndata['feats'] = torch.FloatTensor(dataset.loc[:, ~dataset.columns.isin(cat_cols)].values)
    for c in cat_cols:
        g.ndata[c] = torch.LongTensor(dataset[c].values)


#   config
    config = configparser.ConfigParser()
    config.read(args.config)
    tab_net_params = eval(config[args.config_option]['tab_net'])
    tab_net_params['dims'] = [g.ndata['feats'].shape[-1] + len(cat_cols)*tab_net_params['cat_dims']]+tab_net_params['dims']
    tab_net_params['cats'] = dataset[cat_cols].nunique().to_dict() 
    graph_net_params = eval(config[args.config_option]['graph_net'])
    graph_net_params['n_classes'] = labels.shape[-1]

#   model
    model = SAGE_tabular(tab_net_params, graph_net_params)
    pretrain_path = args.pretrained
    state_dict = torch.load(pretrain_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    summ = count_number_params(model)
    print(f'number of trainable parameters {summ}')

#   dataloader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(graph_net_params['n_layers'])
    dataloader = dgl.dataloading.NodeDataLoader(g, test_mask, sampler, batch_size=10,
            shuffle=True,
            drop_last=True,
            num_workers=0)

#   inference
    result = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm(dataloader, leave=False, total=len(test_mask)//10):
            feats = blocks[0].srcdata['feats']
            cat = {c:blocks[0].srcdata[c] for c in cat_cols}
            out = model(blocks, feats, cat)
            result['pred'] += out.detach().flatten().tolist()
            result['num'] += output_nodes.flatten().tolist()
    result = pd.DataFrame(result)
    result.to_pickle(args.dest)

    
#    result = defaultdict(list)
#    model.eval()
#    with torch.no_grad():
#        blocks = (g for _ in range(graph_net_params['n_layers']))
#        feats = g.srcdata['feats']
#        cat = {c:g.srcdata[c] for c in cat_cols}
#        out = model(blocks, feats, cat)
#        result['pred'] = out.detach().flatten()[val_mask].tolist()
#        result['num'] = val_mask
#    result = pd.DataFrame(result)
#    result.to_pickle(args.dest)
