import dgl
from sklearn import preprocessing
import pandas as pd

def count_number_params(model) :
    summ = 0
    for p in model.parameters():
        if p.requires_grad:
            summ += p.numel()
    return summ

def networkx_to_torch(networkx_graph):
    graph = dgl.from_networkx(networkx_graph)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph

def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)

def replace_na(X, train_mask):
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X
