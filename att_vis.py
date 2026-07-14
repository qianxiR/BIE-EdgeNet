import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def preprocess_attention(attn_tensor: torch.Tensor, H: int, W: int):
    """
    预处理注意力张量：提取单样本 + 还原空间维度
    :param attn_tensor: 注意力权重，shape=(B, H*W, H*W)
    :param H/W: 特征图的高/宽（用于还原维度）
    :param sample_idx: 选择批次中的第几个样本
    :return:
        - attn_flat: 单样本注意力矩阵，shape=(H*W, H*W)
        - attn_spatial: 单样本注意力矩阵（空间维度），shape=(H, W, H, W)
    """
    # 1. 提取单个样本，脱离计算图并转numpy
    attn_flat = attn_tensor.cpu().detach().numpy()  # (H*W, H*W)
    # 2. 还原为空间维度：(H*W, H*W) → (H, W, H, W)
    attn_spatial = attn_flat.reshape(H, W, H, W)
    return attn_flat, attn_spatial


def visualize_global_avg_attention(attn_tensor, H, W):
    """
    可视化全局平均注意力分布（所有目标位置的注意力均值）
    """
    attn_flat, attn_spatial = preprocess_attention(attn_tensor, H, W)

    # 对所有目标位置取平均：(H*W, H*W) → (H*W,) → (H, W)
    global_attn = np.mean(attn_flat, axis=0).reshape(H, W)
    return global_attn