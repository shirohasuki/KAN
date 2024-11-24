from .KAN.KAN import KANLinear
from .KAN.ChebyKAN import ChebyKANLinear
from .KAN.FourierKAN import FourierKANLinear
from .KAN.JacobiKAN import JacobiKANLinear
from .KAN.TaylorKAN import TaylorKANLinear
from .KAN.WaveletKAN import WaveletKANLinear
from .MLP.MLP import MLPLinear
from .CNN.Conv import ConvLinear
from .Transformer.Transformer import TransformerLayer

import torch
import torch.nn as nn

class ModelManager:
    def __init__(self):
        pass  

    def KANLinear(self, in_features, out_features, **kwargs):
        """调用 KANLinear 类"""
        return KANLinear(in_features=in_features, out_features=out_features, **kwargs)

    def ChebyKANLinear(self, in_features, out_features, **kwargs):
        """调用 ChebyKANLinear 类"""
        return ChebyKANLinear(in_features=in_features, out_features=out_features, **kwargs)

    def FourierKANLinear(self, in_features, out_features, **kwargs):
        """调用 FourierKANLinear 类"""
        return FourierKANLinear(in_features=in_features, out_features=out_features, **kwargs)

    def JacobiKANLinear(self, in_features, out_features, **kwargs):
        """调用 JacobiKANLinear 类"""
        return JacobiKANLinear(in_features=in_features, out_features=out_features, **kwargs)

    def TaylorKANLinear(self, in_features, out_features, **kwargs):
        """调用 TaylorKANLinear 类"""
        return TaylorKANLinear(in_features=in_features, out_features=out_features, **kwargs)

    def WaveletKANLinear(self, in_features, out_features, **kwargs):
        """调用 WaveletKANLinear 类"""
        return WaveletKANLinear(in_features=in_features, out_features=out_features, **kwargs)
    
    def MLPLinear(self, in_features, out_features, **kwargs):
        """调用 MLPLinear 类"""
        return MLPLinear(in_features=in_features, out_features=out_features, **kwargs)
    
    def ConvLinear(self, in_channels, out_channels, **kwargs):
        """调用 CNNLayer 类"""
        return ConvLinear(in_channels=in_channels, out_channels=out_channels, **kwargs)
    
    def TransformerLayer(self, embed_dim, num_heads, ff_dim, **kwargs):
        """调用 TransformerLayer 类"""
        return TransformerLayer(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, **kwargs)

    def list_methods(self):
        """列出所有可用的方法"""
        return [method for method in dir(self) if not method.startswith("_")]


if __name__ == "__main__":
    model_manager = ModelManager()

    # 示例调用
    kan_linear = model_manager.KANLinear(in_features=64, out_features=128)
    cheby_kan_linear = model_manager.ChebyKANLinear(in_features=64, out_features=128)
    fourier_kan_linear = model_manager.FourierKANLinear(in_features=64, out_features=128)
    jacobi_kan_linear = model_manager.JacobiKANLinear(in_features=64, out_features=128)
    taylor_kan_linear = model_manager.TaylorKANLinear(in_features=64, out_features=128, order=3)
    wavelet_kan_linear = model_manager.WaveletKANLinear(in_features=64, out_features=128, wavelet_type="mexican_hat")
    mlp_linear = model_manager.MLPLinear(in_features=64, out_features=128)
    conv_linear = model_manager.ConvLinear(in_channels=3, out_channels=16, kernel_size=3,\
        stride=1, padding=1, pool_size=2, pool_stride=2,\
        activation=nn.ReLU, use_batchnorm=True,\
    )
    transformer_layer = model_manager.TransformerLayer(embed_dim=64, num_heads=4, ff_dim=256)

    # 列出所有注册的方法
    print("Available methods:", model_manager.list_methods())

    # 打印实例
    print("KANLinear instance:", kan_linear)
    print("ChebyKANLinear instance:", cheby_kan_linear)
    print("FourierKANLinear instance:", fourier_kan_linear)
    print("JacobiKANLinear instance:", jacobi_kan_linear)
    print("TaylorKANLinear instance:", taylor_kan_linear)
    print("WaveletKANLinear instance:", wavelet_kan_linear)
    print("MLPLinear instance:", mlp_linear)
    print("ConvLinear instance:", conv_linear)
    print("TransformerLayer instance:", transformer_layer)
