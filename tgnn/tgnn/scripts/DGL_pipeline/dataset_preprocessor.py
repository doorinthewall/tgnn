import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import scipy.sparse as sp

def mynormalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = (r_mat_inv.dot(mx)).dot(r_mat_inv)
    return mx

class DataPreprocessor:
    def __init__(self, transformer=None, scaler=None, nan_strategy='mean'):
#        if scaler is None:
#            scaler = StandardScaler()

        if transformer is None:
            transformer = QuantileTransformer()

#        self.scaler = scaler
        self.transformer = transformer
        self.nans = []
        self._nans = []

    def fit(self, dataset):
        self.mean = dataset.mean().to_dict()
#        dataset_ = self.scaler.fit_transform(dataset)
        self.transformer.fit(dataset)

        for col in dataset.columns:
            if dataset.loc[:, col].isna().any():
                self._nans += [col]
                self.nans += [col+'_nan']

        self.col = list(dataset.columns)+self.nans

    def transform(self, dataset):
        nan_feats = []
        for col in self._nans:
            nan_feats += [dataset.loc[:, col].isna().values + 0.] 
        dataset = dataset.fillna(self.mean)
#        dataset = self.scaler.transform(dataset)
        dataset = self.transformer.transform(dataset)
        nan_feats = np.stack(nan_feats, axis=-1)
        dataset = np.concatenate((dataset, nan_feats), axis = -1)
        dataset = pd.DataFrame(dataset, columns=self.col)
        return dataset
