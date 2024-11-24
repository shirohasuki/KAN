import torch
import torch.nn as nn

class MLPLinear(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers=None, activation=nn.ReLU, use_bias=True):
        """
        初始化 MLPLinear 类。
        
        参数:
            in_features (int): 输入特征数。
            out_features (int): 输出特征数。
            hidden_layers (list): 隐藏层的节点数量列表，默认无隐藏层。
            activation (torch.nn.Module): 激活函数类，默认 nn.ReLU。
            use_bias (bool): 是否使用偏置项。
        """
        super(MLPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers or []
        self.activation = activation
        self.use_bias = use_bias

        # 构建网络
        layers = []
        input_dim = in_features
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
            layers.append(self.activation())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_features, bias=use_bias))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (..., in_features)。
        
        返回:
            torch.Tensor: 输出张量，形状为 (..., out_features)。
        """
        return self.network(x)

    def regularization_loss(self, reg_coeff=1e-4):
        """
        计算正则化损失（L2范数）。
        
        参数:
            reg_coeff (float): 正则化系数，默认 1e-4。
        
        返回:
            torch.Tensor: 正则化损失值。
        """
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        return reg_coeff * reg_loss


# 示例用法
if __name__ == "__main__":
    # 创建一个具有两层隐藏层的 MLPLinear 实例
    mlp_linear = MLPLinear(
        in_features=64,
        out_features=10,
        hidden_layers=[128, 64],
        activation=nn.ReLU,
        use_bias=True
    )

    # 打印网络结构
    print(mlp_linear)

    # 输入一个随机张量
    x = torch.randn(32, 64)  # batch_size=32, in_features=64
    output = mlp_linear(x)
    print("Output shape:", output.shape)

    # 计算正则化损失
    reg_loss = mlp_linear.regularization_loss()
    print("Regularization Loss:", reg_loss.item())
