import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLinear(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        pool_size=2,
        pool_stride=2,
        activation=nn.ReLU,
        use_batchnorm=True
    ):
        """
        初始化 CNNLayer 类。
        
        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小，默认 3。
            stride (int): 卷积步幅，默认 1。
            padding (int): 卷积填充，默认 1。
            pool_size (int): 池化窗口大小，默认 2。
            pool_stride (int): 池化步幅，默认 2。
            activation (torch.nn.Module): 激活函数类，默认 nn.ReLU。
            use_batchnorm (bool): 是否使用批归一化，默认 True。
        """
        super(ConvLinear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.activation = activation()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)

    def forward(self, x):
        """
        前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)。
        
        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, new_height, new_width)。
        """
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


# 示例用法
if __name__ == "__main__":
    # 创建一个 CNNLayer 实例
    cnn_layer = ConvLinear(
        in_channels=3,  # 输入通道数 (例如 RGB 图像)
        out_channels=16,  # 输出通道数
        kernel_size=3,  # 卷积核大小
        stride=1,  # 卷积步幅
        padding=1,  # 卷积填充
        pool_size=2,  # 池化窗口大小
        pool_stride=2,  # 池化步幅
        activation=nn.ReLU,  # 激活函数
        use_batchnorm=True  # 使用批归一化
    )

    # 打印网络结构
    print(cnn_layer)

    # 输入一个随机张量 (batch_size=8, in_channels=3, height=32, width=32)
    x = torch.randn(8, 3, 32, 32)
    output = cnn_layer(x)
    print("Output shape:", output.shape)
