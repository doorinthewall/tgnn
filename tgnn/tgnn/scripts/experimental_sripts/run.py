#!/home/Container/envs/bgnn_env/bin/python3
import sys, os
from tgnn.scripts.Models.TGNN import TGNN
import networkx as nx
from tgnn.scripts.data_utils import networkx_to_torch, normalize_features, replace_na, count_number_params
import json
import pandas as pd
import pickle as pk
import configparser
import fire
import torch
from itertools import product
from tensorboardX import SummaryWriter
from sklearn.model_selection import ParameterGrid
from copy import deepcopy
from collections import defaultdict
import numpy as np
import re
import gc
import dgl

def run_model(
                graph,
                config,
                task,
                device,
                config_option,
                dataset,
                labels,
                seed,
                stochastic,
                nepochs,
                masks,
                cat_cols,
                inductive,
                save_folder,
                save_model_outputs,
                ):

    #device
    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #masks
    with open(masks, 'rb') as fp:
        masks = json.load(fp)
        train_mask, val_mask, test_mask = masks[seed]['train'], masks[seed]['val'], masks[seed]['test']

    #data
    dataset = pd.read_csv(dataset)
    if cat_cols is not None:
        with open(cat_cols, 'rb') as fp:
            cat_cols = pk.load(fp)
    else:
        cat_cols = []
    dataset = normalize_features(dataset, train_mask, val_mask, test_mask)
    dataset = replace_na(dataset, train_mask)

    labels = pd.read_csv(labels)
    labels = torch.FloatTensor(labels.values).to(device)

    #graph
    networkx_graph = nx.read_graphml(graph)
    networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
    graph = networkx_to_torch(networkx_graph)
    graph.ndata['feats'] = torch.FloatTensor(dataset.loc[:, ~dataset.columns.isin(cat_cols)].values)
    for c in cat_cols:
        graph.ndata[c] = torch.LongTensor(dataset[c].values)
    X = dataset.values

    #config
    config_ = configparser.ConfigParser(allow_no_value=True)
    config_.read(config)
    config = config_
    tab_net_params = eval(config[config_option]['tab_net'])
    graph_net_params = eval(config[config_option]['graph_net'])
    tab_model_name  = config[config_option]['tab_model_name']
    layer_kwargs = eval(config[config_option]['layer_kwargs'])
    gnn_passes_per_epoch_grid = eval(config[config_option]['gnn_passes_per_epoch'])
    refresh_rounds = config[config_option]['refresh_rounds']
    if refresh_rounds is not None:
        refresh_rounds = int(refresh_rounds)
#    if tab_model_name == 'MGBDT':
#        layer_kwargs['num_boost_round'] = gnn_passes_per_epoch_grid
    batch_size = config.getint(config_option, 'batch_size')
    learning_rate = config.get(config_option, 'learning_rate')
    if learning_rate is not None:
        learning_rate = float(learning_rate)
    accum_steps = config.getint(config_option, 'accum_steps')
    patience = config.getint(config_option, 'patience')
    pretrain = config.getboolean(config_option, 'pretrain')
    pretrain_option = config.get(config_option, 'pretrain_option')

    #hp_grid
    tab_net_params['layer_kwargs'] = list(ParameterGrid(layer_kwargs)) 
    tab_net_params_grid = list(ParameterGrid(tab_net_params))
    graph_net_params_grid = list(ParameterGrid(graph_net_params))
    grid = list(product(graph_net_params_grid, tab_net_params_grid, gnn_passes_per_epoch_grid))
    best_loss = {}
    for graph_net_params, tab_net_params, gnn_passes_per_epoch in grid:
        graph_net_params, tab_net_params = deepcopy(graph_net_params),deepcopy(tab_net_params)

        if tab_model_name != 'MGBDT':
            tab_net_params['dims'] = [graph.ndata['feats'].shape[-1] + len(cat_cols)*tab_net_params['cat_dims']]+tab_net_params['dims']
            tab_net_params['cats'] = dataset[cat_cols].nunique().to_dict() 
        else:
            tab_net_params['dims'] = [graph.ndata['feats'].shape[-1]]+tab_net_params['dims']
        graph_net_params['in_dim'] = tab_net_params.get('nclass', tab_net_params['dims'][-1])
        graph_net_params['out_dim'] = labels.shape[-1]
        sample_nn_size = config.get(config_option, 'sample_nn_size')
        sample_nn_size =  int(sample_nn_size) if sample_nn_size != 'full' else sample_nn_size
        if graph_net_params['name'] in ['gcn', 'agnn', 'appnp']:
            batch_size=1

        #model
        model = TGNN(tab_model_name,
                tab_net_params,
                graph_net_params,
                task,
                gnn_passes_per_epoch,
                stochastic,
                device,
                cat_cols,
                inductive)

        #exp_name
        param_string = tab_model_name + f'-inductive{inductive}'+ '_graph_net' + ''.join([f'-{key}{graph_net_params[key]}' for key in graph_net_params])
        param_string += '_tab_net' + ''.join([f'-{key}{tab_net_params[key]}' for key in tab_net_params if key not in ['layer_kwargs']])
        layer_kwargs = tab_net_params['layer_kwargs']
        param_string += '_layer_kwargs' + ''.join([f'-{key}{layer_kwargs[key]}' for key in layer_kwargs ])
        param_string += f'-passes_per_epoch{gnn_passes_per_epoch}' + f'-stochastic{stochastic}' + f'-patience{patience}'

        exp_name = param_string
        #check_valid
        if inductive:
            if graph_net_params['name'] in ['gcn', 'agnn']:
                continue
        if stochastic:
            if graph_net_params['name'] in ['gcn', 'agnn', 'appnp']:
                continue

        #fit
        writer_path = os.path.join(save_folder, 'logs', str(seed), exp_name)
        if os.path.exists(writer_path):
            raise ValueError(f'{writer_path} exists')
        writer = SummaryWriter(logdir=writer_path, )
        print(exp_name)
        metrics = model.fit(X, labels, graph, train_mask, val_mask, test_mask, 
                batch_size, nepochs, patience=patience,
                metric_name='loss', sample_nn_size=10, initialize=True, writer=writer, learning_rate=learning_rate, 
                accum_steps=accum_steps, pretrain=pretrain, pretrain_option=pretrain_option, refresh_rounds=refresh_rounds)

        writer.close()
        

        #save metrics
        best_loss[exp_name] = min(metrics['loss'], key = lambda x: x[1])
        if save_model_outputs:
            model.model.eval()
            if model.stochastic and (model.tab_model_name != 'MGBDT'):
                with torch.no_grad():
                    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.nlayers)
                    nids = graph.nodes()
                    dataloader = dgl.dataloading.NodeDataLoader(graph, nids, sampler, batch_size=128,
                            shuffle=False,
                            drop_last=False)
                    logits = []
                    for input_nodes, output_nodes, blocks in dataloader:
                        blocks = [b.to(model.device) for b in blocks]
                        feats = blocks[0].srcdata['feats']
                        cat = {c:blocks[0].srcdata[c] for c in model.cat_cols}
                        logits += [model.model(blocks, feats, cat)]
                    logits = torch.cat(logits)

            else:
                if tab_model_name == 'MGBDT':
                    node_feats = torch.FloatTensor(model.tab_net.forward(X)).to(model.device)
                else:
                    node_feats = torch.FloatTensor(X).to(model.device)
                with torch.no_grad():
                    blocks = [graph, graph]
                    logits = model.model(blocks, node_feats)

            if tab_model_name == 'MGBDT':
                outputs = {'gnn':logits.detach().cpu().numpy(), 'tab':node_feats.detach().cpu().numpy()}
            else:
                outputs = {'gnn':logits.detach().cpu().numpy()}

            path = os.path.join(save_folder, seed)
            os.mkdir(path)
            with open(os.path.join(path,'output.pkl'), 'wb') as fp:
                pk.dump(outputs, fp)

        del model
        gc.collect()

    return best_loss

def run(save_folder, graph, config, task, device, config_option,
                                    dataset, labels, seed_num, stochastic, nepochs, masks, cat_cols=None, inductive=False, start_seed=0, save_model_outputs=False):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    loss = {}
    for seed in range(start_seed, seed_num):
        seed = str(seed)
        loss[f'{seed}'] = run_model(graph, config, task, device, config_option, dataset, 
                                        labels, seed, stochastic, nepochs, masks, cat_cols, inductive, save_folder, save_model_outputs)

        if os.path.exists(f'{save_folder}/seed_results.json'):
            with open(f'{save_folder}/seed_results.json', 'r') as f:
                results = json.load(f)
            for seed in loss:
                if seed in results:
                    results[seed].update(loss[seed])
                else:
                    results[seed] = loss[seed]

            loss = results

        with open(f'{save_folder}/seed_results.json', 'w+') as f:
            json.dump(loss, f)

def get_name(name, bgnn_style=False):
    option_list = re.split('_|-', name)
    model_name = option_list[0]
    gcn_name = re.search('(?<=name)\w+', name).group(0)
    inductive = re.search('inductive(?:(?:False)|(?:True))', name).group(0)
    dropout = re.search('(?<=dropout)[0-9\.]+', name).group(0)
    if bgnn_style:
        pre_learning = re.search('(?<=pre_learning)(?:(?:False)|(?:True))', name).group(0) 
        iter_per_epoch = re.search('(?<=iter_per_epoch)[0-9]+', name).group(0)
        lr = re.search('(?<=lr)[0-9\.]+', name).group(0)
        return '_'.join([model_name, gcn_name, dropout, iter_per_epoch, pre_learning, lr])
    else:
#        gnn_passes_per_epoch = re.search('(?<=gnn_passes_per_epoch)[0-9]+', name).group(0)
        dims = re.search('(?<=dims)[0-9\[\] ,]+', name).group(0)
#        stochastic = re.search('stochastic(?:(?:False)|(?:True))', name).group(0)
#        pretrain_option = re.search('(?<=pretrain_option)\w+', name).group(0)
#        pretrain = re.search('(?<=pretrain)(?:(?:False)|(?:True))', name).group(0)
        return '_'.join([model_name,
                            gcn_name,
#                            stochastic,
#                            inductive,
#                            dropout,
#                            gnn_passes_per_epoch,
                            dims,
#                            pretrain_option,
#                            pretrain,
                            ])


def aggregate_results(path, task, filt=None, bgnn_style=False):
    model_best_score = defaultdict(list)
    with open(path, 'r') as fp:
        seed_results = json.load(fp)
    for seed in seed_results:
        model_results_for_seed = defaultdict(list)
        for name, output in seed_results[seed].items():
            if (filt is not None) and (filt not in name):
                continue
            model_name = get_name(name, bgnn_style)
            if bgnn_style:
                model_results_for_seed[model_name] += [output[0]]
            else:
                model_results_for_seed[model_name] += [output]

        for model_name, model_results in model_results_for_seed.items():
            if task == 'regression':
                best_result = min(model_results, key=lambda x: x[1]) # rmse
            else:
                best_result = max(model_results, key=lambda x: x[1]) # accuracy
            model_best_score[model_name].append(best_result[-1])

    
    aggregated = dict()
    for model, scores in model_best_score.items():
        aggregated[model] = (np.mean(scores), np.std(scores))

    return aggregated
            
    

if __name__ == '__main__':
    fire.Fire(run)
