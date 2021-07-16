from tgnn.scripts.DGL_pipeline.sage_model import SAGE
from tgnn.scripts.DGL_pipeline.tab_model import TabModel, ExtendedTabModel
from tgnn.scripts.DGL_pipeline.graph_model import GraphModel
import torch.nn as nn
import torch

class GNN_tabular(nn.Module):
    def __init__(self, 
                 tab_kwargs,
                 sage_kwargs):

        super().__init__()
        self.tab_net = ExtendedTabModel(**tab_kwargs)
        self.mid_act = nn.ReLU()
        self.sage_net = GraphModel(**sage_kwargs)

    def forward(self, blocks, x, cat):
        tab_out = self.tab_net(x, cat)
        tab_out = self.mid_act(tab_out)
        sage_out = self.sage_net(blocks, tab_out)
        return sage_out

    def inference(self, g, x, cat, batch_size):
        tab_out = []
        for i in range(0, len(x), batch_size):
            tab_out += [self.tab_net(x[i:i+batch_size], cat[i:i+batch_size])]
        tab_out = torch.cat(tab_out, dim=0)
        y = self.sage_net.inference(g, tab_out)
        return y

