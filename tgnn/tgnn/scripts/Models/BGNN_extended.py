import itertools
import time
import numpy as np
import torch
import dgl
import sys

from tqdm import tqdm
from collections import defaultdict as ddict
from bgnn.models.BGNN import BGNN
from tgnn.scripts.DGL_pipeline.graph_model import GraphModel
import torch.nn.functional as F

class BGNN_extended(BGNN):
    def __init__(self, stochastic=True, sample_nn_size=10, inductive=False, batch_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlayers = 2
        self.stochastic = stochastic
        self.sample_nn_size = sample_nn_size
        self.inductive = inductive
        self.batch_size = batch_size

    def __name__(self):
        return 'BGNN_extended'

    def init_gnn_model(self):
        self.model = GraphModel(in_dim=self.in_dim,
                                     hidden_dim=self.hidden_dim,
                                     out_dim=self.out_dim,
                                     name=self.name,
                                     dropout=self.dropout, residual = not self.stochastic).to(self.device)

    def networkx_to_torch(self, networkx_graph):
        import dgl
        # graph = dgl.DGLGraph()
        graph = dgl.from_networkx(networkx_graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        return graph

    def train_and_evaluate(self, model_in, target_labels, train_mask, val_mask, test_mask,
                           optimizer, metrics, gnn_passes_per_epoch):
        loss = None
#        train
        for _ in range(gnn_passes_per_epoch):
            loss = self.train_model(model_in, target_labels, train_mask, optimizer)

#        evaluate

        graph, node_feats = model_in
        self.model.eval()
        if self.stochastic:
            with torch.no_grad():
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nlayers)
                nids = graph.nodes()
                dataloader = dgl.dataloading.NodeDataLoader(graph, nids, sampler, batch_size=1024,
                        shuffle=False,
                        drop_last=False)
                logits = []
                for input_nodes, output_nodes, blocks in tqdm(dataloader, leave=False, total=graph.num_nodes()//dataloader.dataloader.batch_size):
                    blocks = [b.to(self.device) for b in blocks]
                    feats = node_feats[input_nodes]
                    logits += [self.model(blocks, feats)]

                logits = torch.cat(logits).squeeze()

        else:
#            if self.tab_model_name == 'MGBDT':
            with torch.no_grad():
                blocks = [graph.to(self.device), graph.to(self.device)]
                logits = self.model(blocks, node_feats).squeeze()


        train_results = self.evaluate_model(logits, target_labels, train_mask)
        val_results = self.evaluate_model(logits, target_labels, val_mask)
        test_results = self.evaluate_model(logits, target_labels, test_mask)
        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                               val_results[metric_name].detach().item(),
                               test_results[metric_name].detach().item()
                               ))
        return train_results['loss'].detach().item()

    def train_model(self, model_in, labels, train_mask, optimizer):
        if self.task == 'regression':
            loss = lambda pred, y: torch.sqrt(F.mse_loss(pred, y))
        elif self.task == 'classification':
            loss = lambda pred, y: F.cross_entropy(pred, y.long())
        else:
            raise NotImplemented("Unknown task. Supported tasks: classification, regression.")

        self.model.train()
        optimizer.zero_grad()
        graph, node_feats  = model_in

        if self.stochastic:
            #dataloader
            sampler = dgl.dataloading.MultiLayerNeighborSampler([self.sample_nn_size]*self.nlayers) 
            if self.inductive:
                train_graph = graph.subgraph(train_mask)
                nids = train_graph.nodes()
            else:
                train_graph = graph
                nids = train_mask

            dataloader = dgl.dataloading.NodeDataLoader(train_graph, nids, sampler,
                                    batch_size=self.batch_size, shuffle=True, drop_last=False)
            batch_size = dataloader.dataloader.batch_size
            
            for nstep, (input_nodes, output_nodes, blocks) in enumerate(tqdm(dataloader, leave=False, total=len(train_mask)//batch_size), 1):
                blocks = [b.to(self.device) for b in blocks]
                if self.inductive:
                    output_nodes = dataloader.collator.g.ndata[dgl.NID][output_nodes]
                    input_nodes = dataloader.collator.g.ndata[dgl.NID][input_nodes]
                feats = node_feats[input_nodes]
                feats_prev  = feats.detach().cpu().clone().data.numpy().copy()
                out = self.model(blocks, feats).squeeze()
                loss_ = loss(out, labels[output_nodes])
                loss_.backward()
#                if self.graph_model_kwargs['name'] in ['gcn', 'agnn', 'appnp']:
#                    continue
                
                optimizer.step()
                assert (feats_prev != node_feats[input_nodes].data.cpu().numpy()).any()
                optimizer.zero_grad()

#            if self.graph_model_kwargs['name'] in ['gcn', 'agnn', 'appnp']:
#                optimizer.step()


        else:
            optimizer.zero_grad()
            if self.inductive:
                graph = graph.subgraph(train_mask)
            blocks = [graph.to(self.device) for _ in range(self.nlayers)]
            if self.inductive:
                out = self.model(blocks, node_feats[graph.ndata[dgl.NID]]).squeeze()
            else:
                out = self.model(blocks, node_feats).squeeze()

            if self.inductive:
                loss_ = loss(out, labels[graph.ndata[dgl.NID]])
            else:
                loss_ = loss(out[train_mask], labels[train_mask])
            loss_.backward()
            optimizer.step()
        return loss_.item()
