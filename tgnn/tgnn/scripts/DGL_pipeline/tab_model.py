import torch.nn as nn
import torch
import os, sys
from tgnn.node.lib.odst import ODST
from tgnn.node.lib.arch import DenseBlock
from copy import copy

class TabModel(nn.Module):
    def __init__(self,
                 dims,
                 cat_dims,
                 cats,
                 nclass,
                 activation = 'relu',
                 dropout = 0.5):

        super().__init__()
        self.dims = dims
        self.nclass = nclass
        self.activation = activation
        self.cat_dims = cat_dims
        self.cats = cats

        if self.activation is None or self.activation == 'relu':
            self.activation = nn.ReLU

        self.dropout = dropout

        self.layers = []
        for i in range(len(dims)-1):
            self.layers += [nn.Linear(dims[i], dims[i+1]), nn.Dropout(dropout), self.activation()]
        self.layers += [nn.Linear(dims[-1], nclass)]
        self.layers = nn.Sequential(*self.layers)
    
        self.embeds = {cat:nn.Embedding(self.cats[cat], self.cat_dims) for cat in self.cats}

        
    def forward(self, x, cat):
        cats = []

        for c in self.cats:
            cats += [self.embeds[c](cat[c])]
        if cats:
            cats = torch.cat(cats, axis=-1)
            x = torch.cat([x, cats], axis=-1)
        return self.layers(x)

class MeanPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


class ExtendedTabModel(nn.Module):
    def __init__(self,
                 dims,
                 cat_dims,
                 cats,
                 nclass,
                 activation = 'relu',
                 layer_type = 'Linear',
                 dropout = 0.5,
                 layer_kwargs=None,
                 ):

        super().__init__()
        self.dims = dims
        self.nclass = nclass
        self.activation = activation
        self.cat_dims = cat_dims
        self.cats = cats

        if self.activation is None or self.activation == 'relu':
            self.activation = nn.ReLU
        
        self.layer_type = layer_type

        if self.layer_type == 'Linear':
            self.layer_type_ = nn.Linear
        elif self.layer_type == 'NODE':
            self.layer_type_ = ODST
        elif self.layer_type == 'Dense':
            self.layer_type_ = DenseBlock
        else:
            raise ValueError(f'{self.__class__.__name__} takes Linear, NODE, Dense as layer_type got {self.layer_type}')

        self.dropout = dropout

        if layer_kwargs is None:
            self.layer_kwargs = {}
        else:
            self.layer_kwargs = layer_kwargs

        self.layers = []
        for i in range(len(dims)-1):
            if self.layer_type == 'Linear':
                init_layer_kwargs = {"in_features":dims[i], "out_features":dims[i+1]} 
            elif self.layer_type == 'NODE':
                init_layer_kwargs = copy(self.layer_kwargs)
                init_layer_kwargs.update({"in_features":dims[i], "tree_dim":dims[i+1]})
            elif self.layer_type == 'Dense':
                init_layer_kwargs = copy(self.layer_kwargs)
                init_layer_kwargs.update({"input_dim":dims[i], "layer_dim":dims[i+1]})


            self.layers += [self.layer_type_(**init_layer_kwargs), nn.Dropout(dropout), self.activation()]

            if self.layer_type == 'NODE':
                self.layers += [MeanPool(dim=-2)]

        if self.layer_type == 'Linear':
            init_layer_kwargs = {"in_features":dims[-1], "out_features":nclass} 
            self.layers += [self.layer_type_(**init_layer_kwargs)]
        elif self.layer_type == 'NODE':
            init_layer_kwargs = copy(self.layer_kwargs)
            init_layer_kwargs.update({"in_features":dims[-1], "tree_dim":nclass})
            self.layers += [self.layer_type_(**init_layer_kwargs)]
            self.layers += [MeanPool(dim=-2)]
        elif self.layer_type == 'Dense':
            layer = self.layers[-3]
            input_dim = layer.num_layers * layer.layer_dim * layer.tree_dim
            self.layers += [nn.Linear(input_dim, nclass)]
        
        self.layers = nn.Sequential(*self.layers)
    
        self.embeds = {cat:nn.Embedding(self.cats[cat], self.cat_dims) for cat in self.cats}

        
    def forward(self, x, cat):
        cats = []

        for c in self.cats:
            cats += [self.embeds[c](cat[c])]
        if cats:
            cats = torch.cat(cats, axis=-1)
            x = torch.cat([x, cats], axis=-1)
        for l in self.layers:
            x = l(x)
        return x
