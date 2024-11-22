# KAN

本仓库收集并整理了多种基于 Kolmogorov-Arnold 网络（KAN）的高效实现，包括 FourierKAN、ChebyKAN、JacobiKAN、TaylorKAN 和 WaveletKAN 等。这些实现旨在提供对不同类型 KAN 模型的深入理解和便捷使用。为了方便观看、阅读和修改，本人基于大量仓库的写法对变种 KAN 进行重构。


## 简介

Kolmogorov-Arnold 网络（KAN）是一类基于 Kolmogorov-Arnold 表示定理的神经网络架构，具有强大的非线性表达能力。本仓库对多种 KAN 的变体进行了实现，包括使用不同基函数（如傅里叶级数、Chebyshev 多项式、Jacobi 多项式、泰勒级数和小波变换）的方法。

## 实现

### KAN

基础的 KAN 实现，使用了 B 样条作为基函数，提供了对 KAN 模型的基本理解。

- 源代码：[KAN.py](KAN.py)

### FourierKAN

使用傅里叶级数作为基函数的 KAN 实现，能够捕捉输入数据的周期性特征。

- 源代码：[FourierKAN.py](FourierKAN.py)

### ChebyKAN

使用 Chebyshev 多项式作为基函数的 KAN 实现，具有良好的数值稳定性和逼近能力。

- 源代码：[ChebyKAN.py](ChebyKAN.py)

### JacobiKAN

使用 Jacobi 多项式作为基函数的 KAN 实现，通过调整参数 \( a \) 和 \( b \)，可以灵活地适应不同的数据分布。

- 源代码：[JacobiKAN.py](JacobiKAN.py)

### TaylorKAN

使用泰勒级数展开作为基函数的 KAN 实现，适用于需要高阶非线性特征的任务。

- 源代码：[TaylorKAN.py](TaylorKAN.py)

### WaveletKAN

使用小波变换作为基函数的 KAN 实现，能够捕捉数据的局部特征或频域特征。

- 源代码：[WaveletKAN.py](WaveletKAN.py)


## Quick Start




## 参考资料

特别感谢以下开源项目对本仓库的支持和贡献：

- [EfficientKAN](https://github.com/Blealtan/efficient-kan)
- [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN)
- [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN/)
- [Wav-KAN](https://github.com/zavareh1/Wav-KAN)
- [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)
- [FourierKAN](https://github.com/GistNoesis/FourierKAN/)

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。