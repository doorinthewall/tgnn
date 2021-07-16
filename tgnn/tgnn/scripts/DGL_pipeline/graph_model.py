from bgnn.models.GNN import GNNModelDGL

class GraphModel(GNNModelDGL):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 dropout=0., name='gat', residual=True, use_mlp=False, join_with_mlp=False):
        
        super().__init__(in_dim, hidden_dim, out_dim,
                 dropout, name, residual, use_mlp, join_with_mlp)

    def forward(self, blocks, features):
        h = features
        if self.name == 'gat':
            h = self.l1(blocks[0], h).flatten(1)
            logits = self.l2(blocks[1], h).mean(1)
        elif self.name in ['appnp']:
            h = self.lin1(h)
            logits = self.l1(blocks[0], h)
        elif self.name == 'agnn':
            h = self.lin1(h)
            h = self.l1(blocks[0], h)
            h = self.l2(blocks[1], h)
            logits = self.lin2(h)
        elif self.name in ['gcn', 'cheb']:
            h = self.drop(h)
            h = self.l1(blocks[0], h)
            logits = self.l2(blocks[1], h)

        return logits
