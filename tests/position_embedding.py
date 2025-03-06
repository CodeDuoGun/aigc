import numpy as np
import torch

def positional_encoding(max_seq_length, embedding_dim):
    """
    生成位置编码矩阵
    :param max_seq_length: 最大序列长度
    :param embedding_dim: 嵌入维度
    :return: 位置编码矩阵，形状为 (max_seq_length, embedding_dim)
    """
    # 创建一个位置编码矩阵
    pos_encoding = np.zeros((max_seq_length, embedding_dim))
    position = np.arange(max_seq_length)[:, np.newaxis]  # (max_seq_length, 1)
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))  # (embedding_dim // 2,)

    # 填充位置编码矩阵
    pos_encoding[:, 0::2] = np.sin(position * div_term)  # 填充偶数位置
    pos_encoding[:, 1::2] = np.cos(position * div_term)  # 填充奇数位置

    pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32)  # 转换为Tensor
    return pos_encoding

# 示例用法
max_seq_length = 50  # 最大序列长度
embedding_dim = 128  # 嵌入维度
pos_enc = positional_encoding(max_seq_length, embedding_dim)
print(pos_enc.shape)  # 输出: torch.Size([50, 128])