import torch
import numpy as np
import cv2
import os


def save_tp_fp_tn_fn_vis(pred_tensor, gt_tensor, img_name, save_root, color_map=None):
    """
    直接传模型输出Tensor和GT Tensor，生成可视化图（无需存中间二值图）
    Args:
        pred_tensor: 模型单张预测结果（Tensor，shape=[H,W]或[1,H,W]，范围0-1（sigmoid后））
        gt_tensor: 单张GT标签（Tensor，shape=[H,W]，值为0（不变）/1（变化））
        img_name: 图像名称
        save_root: 保存根目录
    """
    # 1. 颜色映射（按你的要求：TP白、FP绿、TN黑、FN红）
    if color_map is None:
        color_map = {
            'TP': (255, 255, 255),  # 真阳性（预测对的变化）
            'FP': (253, 251, 115),   # 假阳性（预测错的变化）
            'TN': (0, 0, 0),         # 真阴性（预测对的不变）
            'FN': (0, 0, 255)        # 假阴性（预测错的不变）
        }

    # 2. 处理预测Tensor：转numpy + 二值化（0/255）
    pred_np = pred_tensor.detach().cpu().numpy()
    # 若预测是[1,H,W]，挤压通道维
    if len(pred_np.shape) == 3:
        # 情况1：预测结果为[2, H, W]（二分类，通道0=背景，通道1=前景）
        if pred_np.shape[0] == 2:
            pred_np = pred_np[1, :, :]  # 提取前景（变化区域）的概率图
        # 情况2：预测结果为[1, H, W]（单通道概率图）
        elif pred_np.shape[0] == 1:
            pred_np = pred_np.squeeze(0)
    # 二值化（概率>0.5判定为变化）
    pred_binary = np.where(pred_np > 0.5, 255, 0).astype(np.uint8)

    # 3. 处理GT Tensor：转numpy + 转0/255（核心修正！）
    gt_np = gt_tensor.detach().cpu().numpy()
    # 若GT是[1,H,W]，挤压通道维
    if len(gt_np.shape) == 3:
        # 情况1：预测结果为[2, H, W]（二分类，通道0=背景，通道1=前景）
        if gt_np.shape[0] == 2:
            gt_np = gt_np[1, :, :]  # 提取前景（变化区域）的概率图
        # 情况2：预测结果为[1, H, W]（单通道概率图）
        elif gt_np.shape[0] == 1:
            gt_np = gt_np.squeeze(0)
    # GT从0/1转成0/255（关键！否则gt_binary==255永远不成立）
    gt_binary = np.where(gt_np == 1, 255, 0).astype(np.uint8)

    # 4. 校验尺寸一致
    assert pred_binary.shape == gt_binary.shape, "预测和GT尺寸必须一致！"
    h, w = pred_binary.shape

    # 5. 生成彩色标注图
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    vis_img[:, :, :] = color_map['TN']  # 默认TN（黑色）

    # TP：预测=255 且 GT=255 → 白色
    tp_mask = (pred_binary == 255) & (gt_binary == 255)
    vis_img[tp_mask] = color_map['TP']

    # FP：预测=255 且 GT=0 → 绿色
    fp_mask = (pred_binary == 255) & (gt_binary == 0)
    vis_img[fp_mask] = color_map['FP']

    # FN：预测=0 且 GT=255 → 红色
    fn_mask = (pred_binary == 0) & (gt_binary == 255)
    vis_img[fn_mask] = color_map['FN']

    # 6. 保存
    save_dir = os.path.join(save_root, "tp_fp_tn_fn_vis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{img_name}_tp_fp_tn_fn.png")
    cv2.imwrite(save_path, vis_img)

    return vis_img