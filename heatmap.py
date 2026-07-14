import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
#
#
# def generate_heatmap(feat, method="channel_weight", save_path=None):
#     """
#     生成编码器特征的热力图
#     Args:
#         feat: 单张图像的编码器特征，shape=[32, H, W]（已去除batch维度）
#         method: 可视化方法，可选：
#                 - channel_weight：通道加权（模拟CAM，最优）
#                 - avg_pool：通道平均池化（简单）
#                 - max_pool：通道最大池化（突出强响应区域）
#         save_path: 热力图保存路径（None则返回数组，不保存）
#     Returns:
#         heatmap: 归一化后的热力图数组，shape=[H, W]，值范围0-255（uint8）
#     """
#
#     # if isinstance(feat, torch.Tensor):
#     #     # 处理PyTorch张量：脱离计算图→CPU→转numpy
#     #     feat_2d = feat.detach().cpu().numpy()
#     # elif isinstance(feat, np.ndarray):
#     #     feat_2d = feat
#
#     C = feat.shape[0]
#
#     # # 方法1：通道加权（最优，体现特征整体响应）
#     # if method == "channel_weight":
#     #     # 步骤1：计算每个通道的权重（用通道均值作为权重，模拟CAM）
#     #     channel_weights = torch.mean(feat, dim=(1, 2))  # [32]，每个通道的均值
#     #     channel_weights = F.softmax(channel_weights, dim=0)  # 权重归一化
#     #     # 步骤2：加权求和得到2D特征图
#     #     feat_2d = torch.sum(feat * channel_weights.view(C, 1, 1), dim=0)  # [H,W]
#     #
#     # # 方法2：通道平均池化（简单，体现整体响应）
#     # elif method == "avg_pool":
#     #     feat_2d = torch.mean(feat, dim=0)  # [H,W]
#     #
#     # # 方法3：通道最大池化（突出强响应区域）
#     # elif method == "max_pool":
#     #     feat_2d = torch.max(feat, dim=0)[0]  # [H,W]
#     list = []
#
#     for i in range(C):
#         feat_2d = feat[i, :, :]
#
#         # 归一化到0-255（关键：保证热力图对比度）
#         feat_min = torch.min(feat_2d)
#         feat_max = torch.max(feat_2d)
#         # feat_min = np.min(feat_2d)
#         # feat_max = np.max(feat_2d)
#         feat_norm = (feat_2d - feat_min) / (feat_max - feat_min + 1e-8)  # 0-1
#         heatmap = (feat_norm * 255).cpu().numpy().astype(np.uint8)
#         # heatmap = (feat_norm * 255).astype(np.uint8)
#
#         heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_LINEAR)
#
#         # 应用热力图配色（可选：cv2.COLORMAP_JET是经典热力图配色）
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#         list.append(heatmap)
#
#         # 保存热力图（可选）
#         if save_path is not None:
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             save_path_r = save_path.split(".")[0] + '_{}.jpg'.format(i)
#             cv2.imwrite(save_path_r, heatmap)
#
#     return list
def generate_heatmap(feat, method="channel_weight", save_path=None):
    """
    生成编码器特征的热力图
    Args:
        feat: 单张图像的编码器特征，支持：
                - 3维张量/数组：shape=[C, H, W]（通道数C + 高H + 宽W）
                - 2维张量/数组：shape=[H, W]（自动补通道维度为1）
        method: 可视化方法，可选：
                - channel_weight：通道加权（模拟CAM，最优）
                - avg_pool：通道平均池化（简单）
                - max_pool：通道最大池化（突出强响应区域）
                - per_channel：遍历每个通道生成热力图（你的核心需求）
        save_path: 热力图保存路径（None则返回数组，不保存）
    Returns:
        heatmap_list: 每个通道的热力图数组列表，每个元素shape=[256, 256, 3]（uint8）
    """
    # -------------------------- 核心修复：统一维度为3维（C, H, W） --------------------------
    # 1. 处理张量/数组类型，确保是torch张量（方便统一操作）
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)  # numpy转tensor
    elif not isinstance(feat, torch.Tensor):
        raise TypeError(f"feat必须是torch.Tensor或np.ndarray，当前类型：{type(feat)}")

    # 2. 补全维度：2维→3维（C=1），3维保持不变
    if len(feat.shape) == 2:
        H, W = feat.shape
        feat = feat.unsqueeze(0)  # 2维[H,W] → 3维[1, H, W]
    elif len(feat.shape) == 3:
        C, H, W = feat.shape
    else:
        raise ValueError(f"feat维度必须是2维或3维，当前维度：{len(feat.shape)}")

    # 最终确保feat是3维：[C, H, W]
    C = feat.shape[0]
    heatmap_list = []

    # -------------------------- 遍历每个通道生成热力图（你的核心逻辑） --------------------------
    for i in range(C):
        # 取第i个通道的特征（现在feat是3维，i, :, : 索引合法）
        feat_2d = feat[i, :, :]

        # 归一化到0-255（保证热力图对比度）
        feat_min = torch.min(feat_2d)
        feat_max = torch.max(feat_2d)
        feat_norm = (feat_2d - feat_min) / (feat_max - feat_min + 1e-8)  # 0-1

        # 转为numpy并调整格式
        heatmap = (feat_norm * 255).cpu().numpy().astype(np.uint8)

        # 调整尺寸到256×256（保持你的逻辑）
        heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_LINEAR)

        # 应用热力图配色（JET）
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_list.append(heatmap)

        # 保存每个通道的热力图（保持你的命名逻辑）
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 拼接保存路径：原路径_通道索引.jpg
            save_path_i = f"{os.path.splitext(save_path)[0]}_{i}.jpg"
            cv2.imwrite(save_path_i, heatmap)

    # -------------------------- 保留你注释的其他可视化方法（可选） --------------------------
    # 如果你需要用channel_weight/avg_pool/max_pool方法，取消下面注释即可
    # if method == "channel_weight":
    #     channel_weights = torch.mean(feat, dim=(1, 2))
    #     channel_weights = F.softmax(channel_weights, dim=0)
    #     feat_2d = torch.sum(feat * channel_weights.view(C, 1, 1), dim=0)
    # elif method == "avg_pool":
    #     feat_2d = torch.mean(feat, dim=0)
    # elif method == "max_pool":
    #     feat_2d = torch.max(feat, dim=0)[0]

    return heatmap_list