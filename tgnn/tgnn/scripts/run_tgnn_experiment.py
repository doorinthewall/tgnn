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
from dataset_preprocessor import mynormalize
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import json
import networkx as nx
from data_utils import networkx_to_torch, normalize_features, replace_na, count_number_params
from multiprocessing import cpu_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-s',  '--dataset', type=str, required=True)
    parser.add_argument('--masks', type=str, required=True)
    parser.add_argument('-d', '--dest', type=str, required=True)
    parser.add_argument('-t', '--target', type=str, required=True)
    parser.add_argument('--cat_cols', type=str, required=False)
    parser.add_argument('-g', '--graph', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('--config_option', type=str, required=False)
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--nepochs', type=int, default=35)
    parser.add_argument('--early_stopping_rounds', type=int, default=10, dest='st_rounds')
    parser.add_argument('--task', type=str, default='reg') 
    parser.add_argument('--device', type=str, default='cpu') 

    args = parser.parse_args()
#   writer
    writer = SummaryWriter(f'{args.dest}/logs')

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
    labels = torch.FloatTensor(labels.values).to(args.device)

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
    sample_nn_size = config.getint(args.config_option, 'sample_nn_size')
    batch_size = config.getint(args.config_option, 'batch_size')

#   model
    model = SAGE_tabular(tab_net_params, graph_net_params).to(args.device)
    summ = count_number_params(model)
    print(f'number of trainable parameters {summ}')

    if tab_net_params['layer_type'] is 'NODE':    # hidden initialization
        with torch.no_grad():
            x = g.ndata['feats'][train_mask][:1000].to(args.device)
            cat = {c:g.ndata['c'][train_mask][:1000].to(args.device) for c in cat_cols}
            model.tab_net(x, cat)

#   dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler([sample_nn_size]*graph_net_params['n_layers'])
    dataloader = dgl.dataloading.NodeDataLoader(g, train_mask, sampler, batch_size=batch_size,
            shuffle=True,
            drop_last=True)
#            num_workers=cpu_count())
    
    
    sampler_val = dgl.dataloading.MultiLayerFullNeighborSampler(graph_net_params['n_layers'])
    dataloader_val = dgl.dataloading.NodeDataLoader(g, val_mask, sampler_val, batch_size=10,
            shuffle=True,
            drop_last=True)
#            num_workers=cpu_count())

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
        for input_nodes, output_nodes, blocks in tqdm(dataloader, leave=False, total=len(train_mask)//batch_size):
            blocks = [b.to(torch.device('cuda')) for b in blocks]
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
            for input_nodes, output_nodes, blocks in tqdm(dataloader_val, leave=False, total=len(val_mask)//10):
                blocks = [b.to(torch.device('cuda')) for b in blocks]
                feats = blocks[0].srcdata['feats']
                cat = {c:blocks[0].srcdata[c] for c in cat_cols}
                out = model(blocks, feats, cat)
                loss_ = loss(out, labels[output_nodes])
                val_loss += loss_.item()
                pred += [out.to('cpu')]
                true += [labels[output_nodes].to('cpu')]
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
