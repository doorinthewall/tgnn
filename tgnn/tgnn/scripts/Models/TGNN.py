from bgnn.models.Base import BaseModel
import sys
from tgnn.scripts.DGL_pipeline import sage_tabular_model, graph_tabular_model, graph_model
from tgnn.scripts.DGL_pipeline.sage_tabular_model import SAGE_tabular
from tgnn.scripts.DGL_pipeline.graph_model import GraphModel
from tgnn.scripts.DGL_pipeline.graph_tabular_model import GNN_tabular
import dgl
from collections import defaultdict
import itertools
from tgnn.mGBDT.lib.mgbdt import  MGBDT, MultiXGBModel, LinearModel
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
import pickle as pk

class TGNN(BaseModel):
    def __init__(
            self,
            tab_model_name,
            tab_model_kwargs,
            graph_model_kwargs,
            task,
            gnn_passes_per_epoch,
            stochastic,
            device='cpu',
            cat_cols=None,
            inductive=False,
            accum_steps=1,
            ):
        super(BaseModel, self).__init__()
        self.tab_model_name = tab_model_name
        self.tab_model_kwargs = tab_model_kwargs
        self.graph_model_kwargs = graph_model_kwargs
        self.task = task
        self.nlayers = 2    #magic constant due to bgnn.models.GNN.GNNModelDGL implementation
        if self.graph_model_kwargs['name'] == 'appnp':
            self.nlayers = 1
        self.gnn_passes_per_epoch = gnn_passes_per_epoch if self.tab_model_name == 'MGBDT' else 1
        self.stochastic = stochastic 
        self.device = device
        self.cat_cols = cat_cols if cat_cols else []
        self.inductive = inductive
        self.accum_steps = accum_steps
        
        self.init_tab_net()
        if self.tab_model_name == 'MGBDT':
#            self.tab_net = MGBDT(loss='MSELoss', target_lr=0.5)
#            for i in range(1, len(self.tab_model_kwargs['dims'])):
#                self.tab_net.add_layer(
#                        layer_type='tp_layer', 
#                        F=MultiXGBModel(
#                                        input_size=self.tab_model_kwargs['dims'][i-1], 
#                                        output_size=self.tab_model_kwargs['dims'][i],
#                                        **self.tab_model_kwargs['layer_kwargs'],
#                                        ),
#                        G=None if i == 1 else MultiXGBModel(
#                                        input_size=self.tab_model_kwargs['dims'][i], 
#                                        output_size=self.tab_model_kwargs['dims'][i-1],
#                                        **self.tab_model_kwargs['layer_kwargs'],
#                                        ),
#                        )
            self.model = GraphModel(**self.graph_model_kwargs).to(self.device)
        else:
            self.model = GNN_tabular(self.tab_model_kwargs, self.graph_model_kwargs).to(self.device)


    def init_tab_net(self):
        if self.tab_model_name == 'MGBDT':
            self.tab_net = MGBDT(loss='MSELoss', target_lr=0.5)
            for i in range(1, len(self.tab_model_kwargs['dims'])):
                self.tab_net.add_layer(
                        layer_type='tp_layer', 
                        F=MultiXGBModel(
                                        input_size=self.tab_model_kwargs['dims'][i-1], 
                                        output_size=self.tab_model_kwargs['dims'][i],
                                        **self.tab_model_kwargs['layer_kwargs'],
                                        ),
                        G=None if i == 1 else MultiXGBModel(
                                        input_size=self.tab_model_kwargs['dims'][i], 
                                        output_size=self.tab_model_kwargs['dims'][i-1],
                                        **self.tab_model_kwargs['layer_kwargs'],
                                        ),
                        )
            
            
    
    def fit(self, X, y, graph, train_mask, val_mask, test_mask, 
                batch_size, nepochs, patience=10, metric_name='loss', 
                    sample_nn_size=10, initialize=True, writer=None, learning_rate=None, accum_steps=1,
                    pretrain=False, pretrain_option='stack',refresh_rounds=None):
            
        #early stopping variables
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float('inf')] * 3  # for train/val/test
        best_val_epoch = 0
        epochs_since_last_best_metric = 0

        #initialize_model
        if initialize:
            if pretrain:
                self.initialize_weights(graph, train_mask, target=y, pretrain_option=pretrain_option)
            else:
                self.initialize_weights(graph, train_mask)


        ebar = tqdm(range(nepochs))
        metrics = defaultdict(list)

        #data
        if self.stochastic:
            if (self.graph_model_kwargs['name'] in ['gcn', 'agnn', 'appnp']) or (sample_nn_size == 'full'): 
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nlayers) 
            else:
                sampler = dgl.dataloading.MultiLayerNeighborSampler([sample_nn_size]*self.nlayers) 
            if self.inductive:
                train_graph = graph.subgraph(train_mask)
                nids = train_graph.nodes()
            else:
                train_graph = graph
                nids = train_mask

            dataloader = dgl.dataloading.NodeDataLoader(train_graph, nids, sampler, batch_size=batch_size,
                    shuffle=True,
                    drop_last=False)
            model_in = (dataloader, )
        else:
            node_feats = torch.FloatTensor(X)
            model_in = (node_feats, graph) 
        if self.tab_model_name == 'MGBDT':
            node_feats = self.init_node_features(X)
            if self.stochastic:
                model_in = (node_feats, dataloader, graph)
            else:
                model_in = (node_feats, graph)
            self.update_node_features(node_feats, X)

        #optimizer
        if self.tab_model_name == 'MGBDT':
            if learning_rate is not None:
                optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), (node_feats, )), lr=learning_rate)
            else:
                optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), (node_feats, )))
        else:
            if learning_rate is not None:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            else:
                optimizer = torch.optim.Adam(self.model.parameters())

        #training
        for epoch in ebar:
            
            self.train_and_evaluate(model_in, X, graph, y, train_mask, val_mask, test_mask, optimizer, metrics, accum_steps)
            best_metric, best_val_epoch, epochs_since_last_best_metric \
                    = self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric,
                            metric_name, lower_better=(metric_name not in ['r2', 'accuracy']))
            train_rmse, val_rmse, test_rmse = metrics[metric_name][-1]
            ebar.set_postfix_str(s=f'{metric_name}={metrics[metric_name][-1][1]:0.3f}, best_{metric_name}={best_metric[1]:0.3f} '+
                    f'| Loss {train_rmse:.3f}/{val_rmse:.3f}/{test_rmse:.3f}', refresh=True)
            if patience and epochs_since_last_best_metric > patience:
                break
            if writer is not None:
                for key in metrics:
                    writer.add_scalars(key, {
                                                'train': metrics[key][-1][0],
                                                'val': metrics[key][-1][1],
                                                'test': metrics[key][-1][2]
                                                }, epoch)

            if refresh_rounds is not None:
                if (epoch+1) % refresh_rounds == 0:
                    self.refresh_tab_net(graph, train_mask, node_feats)
                        

        return metrics

    def train_and_evaluate(self, model_in, X, graph, target_labels, train_mask, val_mask, test_mask,
                           optimizer, metrics, accum_steps=1):
        loss = None
#        train
        for _ in range(self.gnn_passes_per_epoch):
            loss = self.train_model(model_in, target_labels, train_mask, optimizer, accum_steps)


#        evaluate
        self.model.eval()
        if self.stochastic and (self.tab_model_name != 'MGBDT'):
            with torch.no_grad():
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nlayers)
                nids = graph.nodes()
                dataloader = dgl.dataloading.NodeDataLoader(graph, nids, sampler, batch_size=100000,
                        shuffle=False,
                        drop_last=False)
                logits = []
                for input_nodes, output_nodes, blocks in tqdm(dataloader, leave=False, total=graph.num_nodes()//dataloader.dataloader.batch_size):
                    blocks = [b.to(self.device) for b in blocks]
                    feats = blocks[0].srcdata['feats']
                    cat = {c:blocks[0].srcdata[c] for c in self.cat_cols}
                    logits += [self.model(blocks, feats, cat)]
                logits = torch.cat(logits)

        else:
            node_feats = model_in[0]
#            if self.tab_model_name == 'MGBDT':
            with torch.no_grad():
                blocks = [graph, graph]
                logits = self.model(blocks, node_feats)

        if self.tab_model_name == 'MGBDT':
            node_feats = model_in[0]
#            import pdb; pdb.set_trace()
            self.tab_net.fit(X[train_mask], node_feats[train_mask].detach().cpu().numpy(), n_epochs=self.gnn_passes_per_epoch)
            self.update_node_features(node_feats, X)
#            if self.stochastic:
#                model_in = (node_feats, *model_in[])
#            else:
#                model_in = (node_feats, graph)


        train_results = self.evaluate_model(logits, target_labels, train_mask)
        val_results = self.evaluate_model(logits, target_labels, val_mask)
        test_results = self.evaluate_model(logits, target_labels, test_mask)
        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                               val_results[metric_name].detach().item(),
                               test_results[metric_name].detach().item()
                               ))
        return loss

    def train_model(self, model_in, labels, train_mask, optimizer, accum_steps=1):
        if self.task == 'regression':
            loss = lambda pred, y: torch.sqrt(F.mse_loss(pred, y))
        elif self.task == 'classification':
            loss = lambda pred, y: F.cross_entropy(pred, y.long())
        else:
            raise NotImplemented("Unknown task. Supported tasks: classification, regression.")

        self.model.train()
        optimizer.zero_grad()


        if self.stochastic:
            if self.tab_model_name == 'MGBDT':
                node_feats, dataloader, graph = model_in
            else:
                dataloader =  model_in[0]
            batch_size = dataloader.dataloader.batch_size
            total = len(train_mask)//batch_size + int((len(train_mask)%batch_size) != 0)
            for nstep, (input_nodes, output_nodes, blocks) in enumerate(tqdm(dataloader, leave=False, total=total), 1):
                blocks = [b.to(self.device) for b in blocks]
                if self.inductive:
                    output_nodes = dataloader.collator.g.ndata[dgl.NID][output_nodes]
                    input_nodes = dataloader.collator.g.ndata[dgl.NID][input_nodes]
                if self.tab_model_name == 'MGBDT':
                    feats = node_feats[input_nodes]
                    out = self.model(blocks, feats)
                else:
                    feats = blocks[0].srcdata['feats']
                    cat = {c:blocks[0].srcdata[c] for c in self.cat_cols}
                    out = self.model(blocks, feats, cat)
                loss_ = loss(out, labels[output_nodes])
                loss_.backward()
#                if self.graph_model_kwargs['name'] in ['gcn', 'agnn', 'appnp']:
#                    continue
                
                if nstep % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if nstep % accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

#            if self.graph_model_kwargs['name'] in ['gcn', 'agnn', 'appnp']:
#                optimizer.step()


        else:
            node_feats, graph = model_in
            optimizer.zero_grad()
            if self.inductive:
                graph = graph.subgraph(train_mask)
            blocks = [graph for _ in range(self.nlayers)]
            if self.tab_model_name == 'MGBDT':
                if self.inductive:
                    out = self.model(blocks, node_feats[graph.ndata[dgl.NID]])
                else:
                    out = self.model(blocks, node_feats)
            else:
                feats = blocks[0].srcdata['feats'].to(self.device)
                cat = {c:blocks[0].srcdata[c].to(self.device) for c in self.cat_cols}
                out = self.model(blocks, feats, cat)

            if self.inductive:
                loss_ = loss(out, labels[graph.ndata[dgl.NID]])
            else:
                loss_ = loss(out[train_mask], labels[train_mask])
            loss_.backward()
            optimizer.step()

    def initialize_weights(self, graph, train_mask, target=None, pretrain_option=None):
        #NODE
        if self.tab_model_name == 'NODE':    # hidden initialization
            with torch.no_grad():
                x = graph.ndata['feats'][train_mask][:1000].to(self.device)
                cat = {c:graph.ndata['c'][train_mask][:1000].to(self.device) for c in self.cat_cols}
                self.model.tab_net(x, cat)

        #MGBDT
        if self.tab_model_name == 'MGBDT':
            feats = graph.ndata['feats'].detach().cpu().numpy()[train_mask]
            self.tab_net.init(feats)
            if target is not None:
                target = target[train_mask]
                if pretrain_option == 'stack':
                    node_feats = self.tab_net.forward(feats)
                    node_feats[:, :target.shape[-1]] = target
                    target_mask = list(range(target.shape[-1]))
                    self.tab_net.fit(feats, node_feats, target_mask=target_mask, n_epochs=self.gnn_passes_per_epoch)
                elif pretrain_option == 'linear':
                    self.tab_net.add_layer('bp_layer', F=LinearModel(self.tab_model_kwargs['dims'][-1], target.shape[-1], loss='MSELoss', learning_rate=0.01))
                    self.tab_net.fit(feats, target.detach().cpu().numpy(), n_epochs=self.gnn_passes_per_epoch)
                    self.tab_net.layers.pop() 
                else:
                    raise ValueError(f'pretrain_option argument got unexpected value: {pretrain_option}')

    def refresh_tab_net(self, graph, train_mask, node_feats, n_rounds=20, n_epochs=3, init_targets=True):
        if self.tab_model_name == 'MGBDT':
            print('refresh')
            feats = graph.ndata['feats'].detach().cpu().numpy()
            if self.inductive:
                feats = feats[train_mask]
            lr=self.tab_model_kwargs['layer_kwargs']['learning_rate']
            depth = self.tab_model_kwargs['layer_kwargs']['max_depth']
            targets = self.tab_net.get_hiddens(feats)
            self.init_tab_net()
            if init_targets:
                self.tab_net.init(feats, targets, n_rounds=n_rounds, max_depth=depth, learning_rate=lr)
            else:
                self.tab_net.init(feats, n_rounds=n_rounds, max_depth=depth, learning_rate=lr)
            self.tab_net.fit(feats[train_mask], node_feats[train_mask].detach().cpu().numpy(), n_epochs=n_epochs*self.gnn_passes_per_epoch)
            self.update_node_features(node_feats, feats)



    def save_model(self, dest):
        if self.tab_model_name == 'MGBDT':
            if not os.path.exists(dest):
                os.mkdir(dest)
            with open(os.path.join(dest, 'tab_model'), 'wb') as fp:
                pk.dump(self.tab_model, fp)
        torch.save(self.model.state_dict(), os.path.join(dest, 'graph_model'))

    def load_model(self, dest):
        if self.tab_model_name == 'MGBDT':
            with open(os.path.join(dest, 'tab_model'), 'rb') as fp:
                tab_net = pk.load(fp)
            state_dict = torch.load(os.path.join(dest, 'graph_model'))
            self.tab_net = tab_net
        self.model.load_state_dict(state_dict, strict=False)

    def init_node_features(self, X):
        node_feats= torch.empty(X.shape[0], self.tab_model_kwargs['dims'][-1], requires_grad=True, device=self.device)
        node_feats.requires_grad=True
        return node_feats
        
    def update_node_features(self, node_features, X):
        node_features.data = torch.FloatTensor(self.tab_net.forward(X))
