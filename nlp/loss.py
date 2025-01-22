
import torch
import torch.nn as nn

def sigmoid():
    pass

def bce_loss():
    """经过sigmoid二元交叉损失"""
    # 创建模型输出和目标标签
    output = torch.sigmoid(torch.randn(3, requires_grad=True))  # 经过Sigmoid激活
    target = torch.empty(3).random_(2).float()  # 生成随机目标标签

    # 创建BCELoss实例
    loss_fn = nn.BCELoss()
    loss = loss_fn(output, target)
    print(loss)


def bce_with_logits_loss():
    """未经过sigmoid的二元交叉损失""" 
    # 创建模型输出和目标标签
    output = torch.randn(3, requires_grad=True)  # 未经过Sigmoid激活
    target = torch.empty(3).random_(2).float()  # 生成随机目标标签

    # 创建BCEWithLogitsLoss实例
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(output, target)
    print(loss)