import torch
import torch.nn as nn

class PCALayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        PCA层初始化
        :param input_dim: 输入特征维度
        :param output_dim: 降维后的特征维度
        """
        super(PCALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 初始化降维矩阵 (output_dim x input_dim)
        self.register_buffer('pca_matrix', torch.eye(output_dim, input_dim))
        self.mean = None  # 保存特征均值
    
    def fit(self, X):
        """
        计算PCA的变换矩阵，类似于sklearn中的fit方法
        :param X: 输入数据 (batch_size, input_dim)
        """
        # 计算均值并去中心化
        self.mean = X.mean(dim=0, keepdim=True)
        X_centered = X - self.mean
        
        # 计算协方差矩阵
        covariance_matrix = torch.mm(X_centered.T, X_centered) / (X.size(0) - 1)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        
        # 按特征值从大到小排序
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 提取前 output_dim 个特征向量
        self.pca_matrix = eigenvectors[:, :self.output_dim].T

    def forward(self, X):
        """
        前向传播，将输入数据映射到降维后的空间
        :param X: 输入数据 (batch_size, input_dim)
        :return: 降维后的数据 (batch_size, output_dim)
        """
        if self.mean is not None:
            X = X - self.mean  # 去中心化
        return torch.mm(X, self.pca_matrix.T)
