import numpy as np
from torch import nn

#--------------------------------------------------------------------------------------------------------------------#
#   Attention机制
#   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
#   然后利用 查询向量query 叉乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
#   然后利用 score 叉乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
#--------------------------------------------------------------------------------------------------------------------#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads  = num_heads
        self.scale      = (dim // num_heads) ** -0.5

        self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C     = x.shape
        qkv         = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v     = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=1), 1)
    return a

Query = np.array([
    [1,0,2],
    [2,2,2],
    [2,1,3]
])

Key = np.array([
    [0,1,1],
    [4,4,0],
    [2,3,1]
])

Value = np.array([
    [1,2,3],
    [2,8,0],
    [2,6,3]
])

scores = Query @ Key.T
print("scores", scores)
scores = soft_max(scores)
print("softmax result", scores)
out = scores @ Value
print("attention score", out)

