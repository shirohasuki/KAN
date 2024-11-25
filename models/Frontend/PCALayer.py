import torch
import torch.nn as nn
from sklearn.decomposition import PCA

class PCALayer:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, data):
        # data: numpy array (N, 28*28)
        return self.pca.fit_transform(data)

    def transform(self, data):
        return self.pca.transform(data)