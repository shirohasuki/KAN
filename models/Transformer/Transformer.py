import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        初始化 TransformerLayer 类。
        
        参数:
            embed_dim (int): 输入嵌入的维度。
            num_heads (int): 多头注意力的头数。
            ff_dim (int): 前馈神经网络的隐藏层维度。
            dropout (float): Dropout 比例，默认 0.1。
        """
        super(TransformerLayer, self).__init__()
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        # 残差连接与 LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (seq_len, batch_size, embed_dim)。
            mask (torch.Tensor): 可选的注意力掩码，形状为 (seq_len, seq_len)。
        
        返回:
            torch.Tensor: 输出张量，形状为 (seq_len, batch_size, embed_dim)。
        """
        # 多头自注意力
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)  # LayerNorm
        
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # 残差连接
        x = self.norm2(x)  # LayerNorm
        
        return x


# 示例用法
if __name__ == "__main__":
    # 输入维度配置
    embed_dim = 64  # 嵌入维度
    num_heads = 4   # 多头注意力头数
    ff_dim = 256    # 前馈网络隐藏层维度
    seq_len = 10    # 序列长度
    batch_size = 32 # 批量大小

    # 创建 TransformerLayer 实例
    transformer_layer = TransformerLayer(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)

    # 打印模型结构
    print(transformer_layer)

    # 测试输入
    x = torch.randn(seq_len, batch_size, embed_dim)  # 输入张量
    mask = None  # 可选的注意力掩码
    output = transformer_layer(x, mask)

    # 打印输出形状
    print("Output shape:", output.shape)
