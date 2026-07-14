import torch
import numpy as np
import cv2
import os


def save_binary_label(pred, img_name, save_root="binary_labels", threshold=0.5):
    """
    将模型预测结果转为背景0、前景255的黑白标签图并保存
    Args:
        pred: 单张图像的模型预测结果，shape=[1, H, W] 或 [2, H, W]（二分类）
        img_name: 图像名称（如"test_001"），用于命名保存的文件
        save_root: 黑白标签图保存的根目录
        threshold: 二值化阈值（0-1），大于阈值为前景（255），否则为背景（0）
    """
    # 步骤1：处理预测结果维度（适配不同输出格式）
    if len(pred.shape) == 3:
        # 情况1：预测结果为[2, H, W]（二分类，通道0=背景，通道1=前景）
        if pred.shape[0] == 2:
            pred = pred[1, :, :]  # 提取前景（变化区域）的概率图
        # 情况2：预测结果为[1, H, W]（单通道概率图）
        elif pred.shape[0] == 1:
            pred = pred.squeeze(0)  # 挤压通道维，变为[H, W]

    # 步骤2：二值化（核心：0=背景，255=前景）
    # 先将tensor转为numpy数组（CPU），再按阈值分割
    pred_np = pred.detach().cpu().numpy()
    binary_label = np.where(pred_np > threshold, 255, 0).astype(np.uint8)  # 关键：255=前景，0=背景

    # 步骤3：创建保存目录（自动创建不存在的文件夹）
    save_dir = os.path.join(save_root, 'pred_result')
    os.makedirs(save_dir, exist_ok=True)

    # 步骤4：保存为PNG格式（无压缩，保证标签值准确）
    save_path = os.path.join(save_dir, f"{img_name}_binary_label.png")
    cv2.imwrite(save_path, binary_label)