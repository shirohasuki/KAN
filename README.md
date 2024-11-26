# KAN Playground

本仓库模块化的实现了多种基于 Kolmogorov-Arnold 网络（KAN）的高效实现。这些实现旨在提供对不同类型 KAN 模型的深入理解和便捷使用。为了方便观看、阅读和修改，本人基于大量仓库的写法对变种 KAN 进行重构。

## 简介

Kolmogorov-Arnold 网络（KAN）是一类基于 Kolmogorov-Arnold 表示定理的神经网络架构，具有强大的非线性表达能力。本仓库对多种 KAN 的改进进行了实现，包括使用不同的损失函数，激活函数，基函数，。

## Quick Start

Step 1. 下载仓库代码
```
git clone https://github.com/shirohasuki/KAN.git
```

Step 2. 安装环境依赖

创建你的conda环境，并执行安装所需的对应包
```
pip install -r requirements.txt
```

Step 3. 点击运行可交互的[demo文件](https://github.com/shirohasuki/KAN/blob/master/demo/demo.ipynb)或[experiments_with_print](https://github.com/shirohasuki/KAN/blob/master/demo/experiments_with_print/)的实验脚本


## Feature
- 开盒即用
- 支持CUDA，兼容GPU 或者 CPU only的环境
- 模型可视化
- 训练可视化
- 结果可视化

## 目前实验中的可调节模块
- 网络结构变换：MLP，CNN，Transformer
- 预处理：PCA，LDA，SVD
- 损失函数：交叉熵，带正则化的交叉熵
- 基函数：B样条，傅里叶级数，切比雪夫多项式，雅可比多项式，泰勒级数展开，小波变换
- 激活函数：SiLU，ReLU，GELU，Mish
- 优化器：LBFGS，Adam，SGD



## 参考资料

特别感谢以下开源项目对本仓库的支持和贡献：

- [pyKAN](https://github.com/KindXiaoming/pykan)
- [EfficientKAN](https://github.com/Blealtan/efficient-kan)
- [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN)
- [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN/)
- [Wav-KAN](https://github.com/zavareh1/Wav-KAN)
- [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)
- [FourierKAN](https://github.com/GistNoesis/FourierKAN/)

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。