import torch.nn.functional
from torchvision import models
from models import CBAM
import math
from torch import nn
import torch.nn.functional as F
from models.ChangeMambaBCD.vmamba import SS2D
from models.submodules import *
from models.re_diffatts import *
from models.MHLA import Diff_MHLA_Normed_Torch, Diff_Window_MHLA_Normed_Torch
from torch.cuda.amp import autocast

class SS2D_CrossAttention_NoDown(nn.Module):
    def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8):
        super().__init__()
        self.heads = heads
        self.dim_per_head = nf // heads
        self.nf = nf
        self.threshold_coeff = threshold_coeff
        assert nf % heads == 0, "nf必须能被heads整除"

        # 保留特征对齐投影层
        def conv_bn(in_channels, out_channels, kernel_size=1):
            groups = min(16, in_channels)
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, groups=groups, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.f1_proj = conv_bn(nf, nf)
        self.f2_proj = conv_bn(nf, nf)
        self.f2_proj_rev = conv_bn(nf, nf)
        self.f1_proj_rev = conv_bn(nf, nf)

        # 保留差异引导分支
        self.diff_proj1 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )
        self.diff_proj2 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )

        # 双向SS2D交叉核心（无下采样，直接处理原始尺寸）
        self.ss2d_cross_f1 = SS2D(
            d_model=nf, d_state=16, ssm_ratio=2.0, dt_rank="auto",
            act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.1,
            initialize="v0", forward_type="v2", channel_first=True
        )
        self.ss2d_cross_f2 = SS2D(
            d_model=nf, d_state=16, ssm_ratio=2.0, dt_rank="auto",
            act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.1,
            initialize="v0", forward_type="v2", channel_first=True
        )

        # 输出投影与残差
        self.out1_proj, self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False), nn.Conv2d(nf, nf, 1, bias=False)
        self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
        self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, f1, f2, diff_feat):
        B, C, H, W = f1.shape
        res1, res2 = f1, f2

        # -------------------------- 1. 差异引导权重计算（直接用原始尺寸） --------------------------
        # 方向1：f2增强f1的差异引导
        diff_proj1 = self.diff_proj1(diff_feat).view(B, self.heads, H*W).unsqueeze(2)
        batch_mean1 = diff_proj1.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold1 = batch_mean1 * self.threshold_coeff
        diff_guide1 = F.relu(diff_proj1 - dynamic_threshold1) + 1e-5
        diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)
        diff_guide1_map = diff_guide1.mean(1).view(B, 1, H, W)  # 原始尺寸的引导权重图

        # 方向2：f1增强f2的差异引导
        diff_proj2 = self.diff_proj2(diff_feat).view(B, self.heads, H*W).unsqueeze(2)
        batch_mean2 = diff_proj2.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold2 = batch_mean2 * self.threshold_coeff
        diff_guide2 = F.relu(diff_proj2 - dynamic_threshold2) + 1e-5
        diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)
        diff_guide2_map = diff_guide2.mean(1).view(B, 1, H, W)

        # -------------------------- 2. 双向SS2D交叉增强（无下采样，直接处理原始特征） --------------------------
        f1_aligned = self.f1_proj(f1)
        f2_aligned = self.f2_proj(f2)
        f2_aligned_rev = self.f2_proj_rev(f2)
        f1_aligned_rev = self.f1_proj_rev(f1)

        # 核心交叉逻辑：原始尺寸特征直接输入SS2D
        cross_f1 = self.ss2d_cross_f1(f1_aligned + f2_aligned * diff_guide1_map)
        cross_f2 = self.ss2d_cross_f2(f2_aligned_rev + f1_aligned_rev * diff_guide2_map)

        # -------------------------- 3. 残差连接（无需要上采样） --------------------------
        attn_weight_f1 = self.out1_proj(cross_f1) + self.residual_scale1 * res1
        attn_weight_f2 = self.out2_proj(cross_f2) + self.residual_scale2 * res2

        return attn_weight_f1, attn_weight_f2

class DW_DifferenceFeatureComplementaryAttention(nn.Module):
    def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8):
        super().__init__()
        self.heads = heads
        self.dim_per_head = nf // heads
        self.nf = nf
        self.threshold_coeff = threshold_coeff
        assert nf % heads == 0, "nf必须能被heads整除"

        # -------------------------- 改进1：深度可分离卷积替换分组卷积（更轻量化） --------------------------
        def conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
            # 深度可分离卷积：depthwise + pointwise，计算量降低≈groups倍
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                          groups=in_channels, bias=False),  # depthwise
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),  # pointwise
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)  # 改进2：加入Dropout提升泛化性
            )

        # 双向投影分支
        self.q1_proj, self.k1_proj, self.v1_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)
        self.q2_proj, self.k2_proj, self.v2_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)

        # -------------------------- 差异引导分支（新增局部上下文增强） --------------------------
        self.diff_proj1 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 3, padding=1, groups=heads, bias=False),  # 局部卷积增强空间连续性
            nn.BatchNorm2d(heads)
        )
        self.diff_proj2 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 3, padding=1, groups=heads, bias=False),
            nn.BatchNorm2d(heads)
        )
        self.diff_context = nn.Conv2d(heads, heads, 3, padding=1, groups=heads, bias=False)  # 差异上下文增强

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.1)  # 改进3：注意力权重Dropout

        # -------------------------- 输出投影与残差（自适应残差缩放） --------------------------
        self.out1_proj, self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False), nn.Conv2d(nf, nf, 1, bias=False)
        self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
        self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

        # -------------------------- 改进4：自适应下采样（替代硬编码字典） --------------------------
        self.min_down_size = 8  # 下采样后最小尺寸

        # 改进5：可学习温度系数（替代静态scale）
        self.temp = nn.Parameter(torch.tensor(1.0))

        # 改进6：多头动态融合权重
        self.head_weights = nn.Parameter(torch.ones(self.heads))

    def get_adaptive_down_ratio(self, H):
        """根据输入尺寸自适应计算下采样比例"""
        base_ratio = 4
        ratio = min(base_ratio, H // self.min_down_size)
        return max(1, ratio)

    def forward(self, f1, f2, diff_feat):
        B, C, H, W = f1.shape
        res1, res2 = f1, f2

        # 1. 自适应下采样
        down_ratio = self.get_adaptive_down_ratio(H)
        if down_ratio > 1:
            f1_down = F.avg_pool2d(f1, kernel_size=down_ratio, stride=down_ratio)
            f2_down = F.avg_pool2d(f2, kernel_size=down_ratio, stride=down_ratio)
            diff_feat_down = F.avg_pool2d(diff_feat, kernel_size=down_ratio, stride=down_ratio)
            h_down, w_down = f1_down.shape[2], f1_down.shape[3]
            hw_down = h_down * w_down
        else:
            f1_down, f2_down, diff_feat_down = f1, f2, diff_feat
            h_down, w_down = H, W
            hw_down = H * W

        # 2. 差异引导权重计算（改进：分位数阈值+上下文增强）
        # 方向1：f2增强f1的差异引导
        diff_proj1 = self.diff_proj1(diff_feat_down)  # (B, heads, h_down, w_down)
        diff_proj1 = self.diff_context(diff_proj1)  # 局部上下文增强
        diff_proj1 = diff_proj1.view(B, self.heads, hw_down).unsqueeze(2)  # (B, heads, 1, hw_down)

        # 改进：分位数阈值（抗噪声，替代均值阈值）

        with autocast(enabled=False):
            diff_proj1_32 = diff_proj1.to(dtype=torch.float32)
            quantile_90 = torch.quantile(diff_proj1_32, 0.9, dim=-1, keepdim=True)
        # quantile_90 = torch.quantile(diff_proj1, 0.9, dim=-1, keepdim=True)
        dynamic_threshold1 = quantile_90 * self.threshold_coeff

        # 区分正负差异
        diff_pos = F.relu(diff_proj1 - dynamic_threshold1)
        diff_neg = F.relu(dynamic_threshold1 - diff_proj1)
        diff_alpha = nn.Parameter(torch.tensor(0.7, device=f1.device))  # 可学习正差异权重
        diff_guide1 = diff_alpha * diff_pos + (1 - diff_alpha) * diff_neg
        diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)

        # 方向2：f1增强f2的差异引导（对称逻辑）
        diff_proj2 = self.diff_proj2(diff_feat_down)
        diff_proj2 = self.diff_context(diff_proj2)
        diff_proj2 = diff_proj2.view(B, self.heads, hw_down).unsqueeze(2)


        with autocast(enabled=False):
            diff_proj2_32 = diff_proj2.to(dtype=torch.float32)
            quantile_90_2 = torch.quantile(diff_proj2_32, 0.9, dim=-1, keepdim=True)
        # quantile_90_2 = torch.quantile(diff_proj2, 0.9, dim=-1, keepdim=True)
        dynamic_threshold2 = quantile_90_2 * self.threshold_coeff

        diff_pos2 = F.relu(diff_proj2 - dynamic_threshold2)
        diff_neg2 = F.relu(dynamic_threshold2 - diff_proj2)
        diff_guide2 = diff_alpha * diff_pos2 + (1 - diff_alpha) * diff_neg2
        diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Q/K/V投影与注意力计算
        # 方向1：f2增强f1的注意力
        Q1 = self.q1_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)
        K1 = self.k1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)
        V1 = self.v1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)

        # 改进：可学习温度系数的动态缩放
        scale_factor = torch.sqrt(torch.tensor(hw_down, dtype=torch.float32, device=f1.device))
        dynamic_scale = (self.dim_per_head ** -0.5) * (scale_factor ** 0.5) * self.temp
        score1 = torch.matmul(Q1, K1) * dynamic_scale
        score1 = score1 * diff_guide1

        # 注意力加权（加入Dropout）
        weight1 = self.softmax(score1)
        weight1 = self.attn_drop(weight1)  # 正则化注意力权重

        out1 = torch.matmul(V1, weight1.transpose(-1, -2))
        # 改进：多头动态融合
        out1 = out1 * self.head_weights.view(1, self.heads, 1, 1)
        out1 = out1.view(B, self.heads * self.dim_per_head, h_down, w_down)

        # 改进：可学习上采样（替代双线性插值）
        if down_ratio > 1:
            out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=True)
        # 自适应残差缩放（基于特征方差）
        res1_var = torch.var(res1, dim=(2, 3), keepdim=True)
        attn_weight_f1 = self.out1_proj(out1) + (self.residual_scale1 * res1_var) * res1

        # 方向2：f1增强f2的注意力（对称逻辑）
        Q2 = self.q2_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)
        K2 = self.k2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)
        V2 = self.v2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)

        score2 = torch.matmul(Q2, K2) * dynamic_scale
        score2 = score2 * diff_guide2

        weight2 = self.softmax(score2)
        weight2 = self.attn_drop(weight2)

        out2 = torch.matmul(V2, weight2.transpose(-1, -2))
        out2 = out2 * self.head_weights.view(1, self.heads, 1, 1)
        out2 = out2.view(B, C, h_down, w_down)

        if down_ratio > 1:
            out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=True)
        res2_var = torch.var(res2, dim=(2, 3), keepdim=True)
        attn_weight_f2 = self.out2_proj(out2) + (self.residual_scale2 * res2_var) * res2

        return attn_weight_f1, attn_weight_f2

class LightWeight_DifferenceFeatureComplementaryAttention(nn.Module):
    def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8):
        super().__init__()
        self.heads = heads
        self.dim_per_head = nf // heads
        self.nf = nf
        self.threshold_coeff = threshold_coeff
        assert nf % heads == 0, "nf必须能被heads整除"

        # -------------------------- 轻量化投影层（保留分组卷积，降低计算量） --------------------------
        def conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
            groups = min(16, in_channels)  # 分组卷积轻量化
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                          groups=groups, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 双向投影分支（输出通道仍为nf，分注意力头）
        self.q1_proj, self.k1_proj, self.v1_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)
        self.q2_proj, self.k2_proj, self.v2_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)

        # -------------------------- 差异引导分支（输出通道=heads，与原逻辑一致） --------------------------
        self.diff_proj1 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)  # (B, heads, H, W)
        )
        self.diff_proj2 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        # -------------------------- 输出投影与残差（保留原逻辑） --------------------------
        self.out1_proj, self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False), nn.Conv2d(nf, nf, 1, bias=False)
        self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
        self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

        # -------------------------- 新增：动态下采样比例映射（适配多尺度输入） --------------------------
        # 输入空间尺寸 → 下采样比例（小尺寸特征不下采样，避免信息丢失）
        self.downsample_ratios = {
            64: 4,   # 64×64 → 16×16（4倍下采样）
            32: 4,   # 32×32 → 8×8（4倍下采样）
            16: 2,   # 16×16 → 8×8（2倍下采样）
            8: 1     # 8×8 → 8×8（不下采样）
        }

    def forward(self, f1, f2, diff_feat):
        B, C, H, W = f1.shape  # 输入尺寸：(1, 64, 64, 64) → H=64, W=64；以此类推
        res1, res2 = f1, f2    # 残差保留原始特征

        # -------------------------- 1. 动态下采样（核心优化：根据输入尺寸调整） --------------------------
        # 获取当前特征的下采样比例（默认1，即不下采样）
        down_ratio = self.downsample_ratios.get(H, 1)
        if down_ratio > 1:
            # 下采样特征（使用平均池化，保留全局信息）
            f1_down = F.avg_pool2d(f1, kernel_size=down_ratio, stride=down_ratio)  # (B, C, H/ratio, W/ratio)
            f2_down = F.avg_pool2d(f2, kernel_size=down_ratio, stride=down_ratio)
            diff_feat_down = F.avg_pool2d(diff_feat, kernel_size=down_ratio, stride=down_ratio)
            h_down, w_down = f1_down.shape[2], f1_down.shape[3]  # 下采样后的尺寸
            hw_down = h_down * w_down  # 下采样后的空间元素数
        else:
            # 小尺寸特征（如8×8）不下采样，直接使用原始特征
            f1_down, f2_down, diff_feat_down = f1, f2, diff_feat
            h_down, w_down = H, W
            hw_down = H * W

        # -------------------------- 2. 差异引导权重计算（基于下采样特征，显存友好） --------------------------
        # 方向1：f2增强f1的差异引导
        # 下采样特征投影 → 展平为 (B, heads, hw_down) → 增加维度用于广播
        diff_proj1 = self.diff_proj1(diff_feat_down).view(B, self.heads, hw_down).unsqueeze(2)  # (B, heads, 1, hw_down)
        # 动态阈值（基于下采样后的特征均值）
        batch_mean1 = diff_proj1.mean(dim=(-1, -2), keepdim=True)  # (B, heads, 1, 1)
        dynamic_threshold1 = batch_mean1 * self.threshold_coeff
        # 过滤弱关联并归一化
        diff_guide1 = F.relu(diff_proj1 - dynamic_threshold1) + 1e-5
        diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)

        # 方向2：f1增强f2的差异引导（对称逻辑）
        diff_proj2 = self.diff_proj2(diff_feat_down).view(B, self.heads, hw_down).unsqueeze(2)
        batch_mean2 = diff_proj2.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold2 = batch_mean2 * self.threshold_coeff
        diff_guide2 = F.relu(diff_proj2 - dynamic_threshold2) + 1e-5
        diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)

        # -------------------------- 3. Q/K/V投影与注意力计算（下采样特征上操作） --------------------------
        # 方向1：f2增强f1的注意力
        # Q1: 下采样特征投影 → 重塑为 (B, heads, hw_down, dim_per_head)
        Q1 = self.q1_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)  # (B, heads, hw_down, dim)
        # K1: 下采样特征投影 → (B, heads, dim_per_head, hw_down)
        K1 = self.k1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)
        # V1: 下采样特征投影 → (B, heads, dim_per_head, hw_down)
        V1 = self.v1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)

        # 注意力分数计算（基于下采样尺寸，矩阵尺寸大幅减小）
        scale_factor = torch.sqrt(torch.tensor(hw_down, dtype=torch.float32, device=f1.device))
        dynamic_scale = (self.dim_per_head ** -0.5) * (scale_factor ** 0.5)
        score1 = torch.matmul(Q1, K1) * dynamic_scale  # (B, heads, hw_down, hw_down)
        score1 = score1 * diff_guide1  # 应用差异引导

        # 注意力加权与上采样恢复
        weight1 = self.softmax(score1)  # (B, heads, hw_down, hw_down)
        out1 = torch.matmul(V1, weight1.transpose(-1, -2))  # (B, heads, dim_per_head, hw_down)
        out1 = out1.view(B, self.heads * self.dim_per_head, h_down, w_down)  # 重塑为特征图
        # 上采样回原始尺寸（与输入f1同形）
        out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=True)
        # 输出投影 + 残差连接
        attn_weight_f1 = self.out1_proj(out1) + self.residual_scale1 * res1

        # -------------------------- 4. 方向2：f1增强f2的注意力（对称逻辑） --------------------------
        Q2 = self.q2_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)
        K2 = self.k2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)
        V2 = self.v2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)

        score2 = torch.matmul(Q2, K2) * dynamic_scale
        score2 = score2 * diff_guide2

        weight2 = self.softmax(score2)
        out2 = torch.matmul(V2, weight2.transpose(-1, -2)).view(B, C, h_down, w_down)
        out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=True)  # 上采样恢复
        attn_weight_f2 = self.out2_proj(out2) + self.residual_scale2 * res2

        return attn_weight_f1, attn_weight_f2


class DiffAttention(nn.Module):
    """适配BIE的差分注意力模块（纯重叠窗口局部差分注意力，无下采样/上采样）"""

    def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8, depth=1, window_size=8, overlap_ratio=0.5):
        super().__init__()

        self.heads = heads
        self.dim_per_head = nf // heads
        self.nf = nf
        self.threshold_coeff = threshold_coeff
        self.depth = depth
        self.window_size = window_size  # 核心：局部窗口尺寸
        self.overlap_ratio = overlap_ratio  # 核心：窗口重叠比例
        assert nf % heads == 0, "nf必须能被heads整除"

        # -------------------------- 1. 重叠窗口配置（核心新增） --------------------------
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size  # 重叠窗口尺寸
        self.pad = (self.overlap_win_size - self.window_size) // 2  # 填充值（保证窗口对齐）
        # 重叠窗口展开层（将特征划分为重叠窗口）
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=self.window_size,
            padding=self.pad
        )
        # 🔥 移除错误的Buffer注册（OCDA里没有这行！）
        # self.register_buffer("win_count", torch.tensor(0))

        # -------------------------- 2. 轻量化Q/K/V投影（保留分组卷积） --------------------------
        def conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
            groups = min(16, in_channels)
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 双向Q/K/V投影（f1→f2 方向 + f2→f1 方向）
        self.q1_proj = conv_bn(nf, nf)  # f1的Q
        self.k1_proj = conv_bn(nf, nf)  # f2的K
        self.v1_proj = conv_bn(nf, nf)  # f2的V
        self.q2_proj = conv_bn(nf, nf)  # f2的Q
        self.k2_proj = conv_bn(nf, nf)  # f1的K
        self.v2_proj = conv_bn(nf, nf)  # f1的V

        # -------------------------- 3. 差分注意力核心参数（复用OCDA逻辑） --------------------------
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * self.depth)
        # 可学习的λ参数（每个注意力头独立）
        self.lambda_q1 = nn.Parameter(
            torch.zeros(heads, self.dim_per_head, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(heads, self.dim_per_head, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(heads, self.dim_per_head, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(heads, self.dim_per_head, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        # -------------------------- 4. 差异引导分支（保留BIE的空间/通道差异逻辑） --------------------------
        self.diff_proj1 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )
        self.diff_proj2 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )

        # -------------------------- 5. 其他配置 --------------------------
        self.softmax = nn.Softmax(dim=-1)
        self.out1_proj = nn.Conv2d(nf, nf, 1, bias=False)
        self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False)
        self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
        self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

    def _get_window_info(self, x):
        """计算窗口数量和特征尺寸信息（适配任意输入尺寸）"""
        B, C, H, W = x.shape
        # 计算非重叠窗口数量（H/W方向）
        num_h_win = (H + self.window_size - 1) // self.window_size
        num_w_win = (W + self.window_size - 1) // self.window_size
        num_total_win = num_h_win * num_w_win  # 总窗口数
        # 重叠窗口内的元素数
        win_elem_num = self.overlap_win_size * self.overlap_win_size
        return num_total_win, win_elem_num, H, W

    def _fold_windows(self, x, win_count, H, W):
        """将重叠窗口特征折叠回原始尺寸（逆Unfold操作）→ 🔥 新增win_count参数，对齐OCDA局部变量逻辑"""
        B = x.shape[0] // win_count
        C = self.nf
        win_elem_num = self.overlap_win_size * self.overlap_win_size
        # 重塑为Unfold的输出格式：(B, C*win_elem_num, num_total_win)
        x = rearrange(x, '(b nw) ne c -> b (c ne) nw', b=B, nw=win_count, ne=win_elem_num)
        # 折叠回原始尺寸
        x = F.fold(
            x,
            output_size=(H, W),
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=self.window_size,
            padding=self.pad
        )
        # 归一化（解决重叠区域的像素值重复累加问题）
        count_mat = torch.ones((1, 1, H, W), device=x.device)
        count_mat = self.unfold(count_mat)
        count_mat = F.fold(
            count_mat,
            output_size=(H, W),
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=self.window_size,
            padding=self.pad
        )
        x = x / (count_mat + 1e-6)  # 避免除以0
        return x

    def forward(self, f1, f2, diff_feat):
        B, C, H, W = f1.shape
        res1, res2 = f1, f2

        # -------------------------- 1. 重叠窗口展开（核心：无下采样，直接划窗） --------------------------
        # 🔥 改为局部变量（对齐OCDA的I变量逻辑），不再赋值给self.win_count
        win_count, win_elem_num, _, _ = self._get_window_info(f1)
        # 对Q/K/V和差异特征做重叠窗口展开
        # 展开格式：(B, C*win_elem_num, num_total_win) → 重塑为：(B*num_total_win, win_elem_num, C)
        f1_unfold = self.unfold(f1)
        f2_unfold = self.unfold(f2)
        diff_feat_unfold = self.unfold(diff_feat)

        f1_win = rearrange(f1_unfold, 'b (c ne) nw -> (b nw) ne c', ne=win_elem_num, c=C, nw=win_count)
        f2_win = rearrange(f2_unfold, 'b (c ne) nw -> (b nw) ne c', ne=win_elem_num, c=C, nw=win_count)
        diff_feat_win = rearrange(diff_feat_unfold, 'b (c ne) nw -> (b nw) ne c', ne=win_elem_num, c=C, nw=win_count)

        # -------------------------- 2. 差异引导权重（基于重叠窗口，局部差异建模） --------------------------
        # 方向1：f2增强f1的差异引导（每个窗口独立计算）
        diff_proj1 = self.diff_proj1(rearrange(diff_feat_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))
        diff_proj1 = rearrange(diff_proj1, 'b heads nw ne -> (b nw) heads ne').unsqueeze(2)  # (B*nw, heads, 1, ne)
        batch_mean1 = diff_proj1.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold1 = batch_mean1 * self.threshold_coeff
        diff_guide1 = F.relu(diff_proj1 - dynamic_threshold1) + 1e-5
        diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)

        # 方向2：f1增强f2的差异引导（对称逻辑）
        diff_proj2 = self.diff_proj2(rearrange(diff_feat_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))
        diff_proj2 = rearrange(diff_proj2, 'b heads nw ne -> (b nw) heads ne').unsqueeze(2)
        batch_mean2 = diff_proj2.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold2 = batch_mean2 * self.threshold_coeff
        diff_guide2 = F.relu(diff_proj2 - dynamic_threshold2) + 1e-5
        diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)

        # -------------------------- 3. 差分注意力计算（方向1：f2增强f1） --------------------------
        # Q/K/V投影（每个窗口独立投影）
        Q1 = self.q1_proj(rearrange(f1_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))
        K1 = self.k1_proj(rearrange(f2_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))
        V1 = self.v1_proj(rearrange(f2_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))

        # 重塑为注意力维度：(B*nw, heads, ne, dim_per_head)
        Q1 = rearrange(Q1, 'b (heads dim) nw ne -> (b nw) heads ne dim', heads=self.heads, dim=self.dim_per_head, nw=win_count)
        K1 = rearrange(K1, 'b (heads dim) nw ne -> (b nw) heads dim ne', heads=self.heads, dim=self.dim_per_head, nw=win_count)
        V1 = rearrange(V1, 'b (heads dim) nw ne -> (b nw) heads dim ne', heads=self.heads, dim=self.dim_per_head, nw=win_count)

        # 注意力缩放因子（适配窗口尺寸）
        scale_factor = torch.sqrt(torch.tensor(win_elem_num, dtype=torch.float32, device=f1.device))
        dynamic_scale = (self.dim_per_head ** -0.5) * (scale_factor ** 0.5)

        # 差分注意力核心：attn1 - λ*attn2
        score1_1 = torch.matmul(Q1, K1) * dynamic_scale  # 主注意力分数
        score1_2 = torch.matmul(Q1, K1) * dynamic_scale * 0.5  # 辅助注意力分数（平衡差分效应）

        # λ计算（可学习 + 深度自适应）→ 对齐OCDA的lambda_full广播逻辑
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(Q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(Q1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        lambda_full = lambda_1.unsqueeze(0).repeat(B * win_count, 1).unsqueeze(-1).unsqueeze(-1)

        # 融合差异引导 + 软归一化
        score1 = score1_1 - lambda_full * score1_2
        score1 = score1 * diff_guide1  # 局部差异引导注意力焦点
        weight1 = self.softmax(score1)

        # 注意力加权（每个窗口独立加权）
        out1 = torch.matmul(V1, weight1.transpose(-1, -2))
        # 重塑为窗口特征格式：(B*nw, ne, C)
        out1 = rearrange(out1, '(b nw) heads dim ne -> (b nw) ne (heads dim)', b=B, nw=win_count, heads=self.heads, dim=self.dim_per_head)

        # -------------------------- 4. 窗口折叠回原始尺寸（无下采样，直接折叠） --------------------------
        out1 = self._fold_windows(out1, win_count, H, W)  # 传入局部变量win_count
        # 输出投影 + 残差连接（原始特征尺寸）
        attn_weight_f1 = self.out1_proj(out1) + self.residual_scale1 * res1

        # -------------------------- 5. 差分注意力计算（方向2：f1增强f2，对称逻辑） --------------------------
        Q2 = self.q2_proj(rearrange(f2_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))
        K2 = self.k2_proj(rearrange(f1_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))
        V2 = self.v2_proj(rearrange(f1_win, '(b nw) ne c -> b c nw ne', b=B, nw=win_count))

        Q2 = rearrange(Q2, 'b (heads dim) nw ne -> (b nw) heads ne dim', heads=self.heads, dim=self.dim_per_head, nw=win_count)
        K2 = rearrange(K2, 'b (heads dim) nw ne -> (b nw) heads dim ne', heads=self.heads, dim=self.dim_per_head, nw=win_count)
        V2 = rearrange(V2, 'b (heads dim) nw ne -> (b nw) heads dim ne', heads=self.heads, dim=self.dim_per_head, nw=win_count)

        score2_1 = torch.matmul(Q2, K2) * dynamic_scale
        score2_2 = torch.matmul(Q2, K2) * dynamic_scale * 0.5
        score2 = score2_1 - lambda_full * score2_2
        score2 = score2 * diff_guide2
        weight2 = self.softmax(score2)

        out2 = torch.matmul(V2, weight2.transpose(-1, -2))
        out2 = rearrange(out2, '(b nw) heads dim ne -> (b nw) ne (heads dim)', b=B, nw=win_count, heads=self.heads, dim=self.dim_per_head)

        # 窗口折叠 + 输出投影 + 残差
        out2 = self._fold_windows(out2, win_count, H, W)  # 传入局部变量win_count
        attn_weight_f2 = self.out2_proj(out2) + self.residual_scale2 * res2

        return attn_weight_f1, attn_weight_f2



class MHLA_DifferenceFeatureComplementaryAttention(nn.Module):
    def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8, feature_size=16, sr_ratio=1):
        super().__init__()
        self.heads = heads
        self.dim_per_head = nf // heads
        self.nf = nf
        self.threshold_coeff = threshold_coeff
        assert nf % heads == 0, "nf必须能被heads整除"

        # -------------------------- 1. 保留原轻量化投影层（分组卷积） --------------------------
        def conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
            groups = min(16, in_channels)  # 分组卷积轻量化
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                          groups=groups, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 双向QKV投影（保留原逻辑，适配MHLA输入）
        self.q1_proj, self.k1_proj, self.v1_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)
        self.q2_proj, self.k2_proj, self.v2_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)

        # -------------------------- 2. 保留原差异引导分支 --------------------------
        self.diff_proj1 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )
        self.diff_proj2 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )

        self.sr_ratio = sr_ratio
        self.embed_len = feature_size * feature_size
        base_window = self.embed_len // (self.sr_ratio ** 2) if self.sr_ratio > 1 else self.embed_len
        self.window_size = int(round(math.sqrt(base_window))) ** 2

        self.mhla1 = Diff_Window_MHLA_Normed_Torch(
            dim=nf,
            heads=heads,
            dim_head=self.dim_per_head,
            dropout=0.1,
            qkv_bias=True,
            embed_len=self.embed_len,
            window_size=self.window_size,
            transform="cos",
            local_thres=1.5,
            exp_sigma=3
        )
        self.mhla2 = Diff_Window_MHLA_Normed_Torch(
            dim=nf,
            heads=heads,
            dim_head=self.dim_per_head,
            dropout=0.1,
            qkv_bias=True,
            embed_len=self.embed_len,
            window_size=self.window_size,
            transform="cos",
            local_thres=1.5,
            exp_sigma=3
        )

        # -------------------------- 4. 保留原输出投影与残差 --------------------------
        self.out1_proj, self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False), nn.Conv2d(nf, nf, 1, bias=False)
        self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
        self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

        # -------------------------- 5. 保留原动态下采样比例映射 --------------------------
        # self.downsample_ratios = {
        #     64: 4,   # 64×64 → 16×16
        #     32: 4,   # 32×32 → 8×8
        #     16: 2,   # 16×16 → 8×8
        #     8: 1     # 8×8 → 8×8
        # }

    def forward(self, f1, f2, diff_feat):
        B, C, H, W = f1.shape
        res1, res2 = f1, f2

        f1_down, f2_down, diff_feat_down = f1, f2, diff_feat
        h_down, w_down = H, W
        hw_down = H * W

        # -------------------------- 2. 保留原差异引导权重计算 --------------------------
        # 方向1：f2增强f1的差异引导
        diff_proj1 = self.diff_proj1(diff_feat_down).view(B, self.heads, hw_down).unsqueeze(2)
        batch_mean1 = diff_proj1.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold1 = batch_mean1 * self.threshold_coeff
        diff_guide1 = F.relu(diff_proj1 - dynamic_threshold1) + 1e-5
        diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)
        diff_guide1_map = diff_guide1.view(B, self.heads, H, W).mean(dim=1, keepdim=True)

        # 方向2：f1增强f2的差异引导
        diff_proj2 = self.diff_proj2(diff_feat_down).view(B, self.heads, hw_down).unsqueeze(2)
        batch_mean2 = diff_proj2.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold2 = batch_mean2 * self.threshold_coeff
        diff_guide2 = F.relu(diff_proj2 - dynamic_threshold2) + 1e-5
        diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)
        diff_guide2_map = diff_guide2.view(B, self.heads, H, W).mean(dim=1, keepdim=True)

        window_size = self.window_size
        embed_len = self.embed_len
        N_pieces =  embed_len // window_size

        # -------------------------- 3. 核心替换：MHLA注意力计算（方向1：f2增强f1） --------------------------
        # Step1: QKV投影（保留原轻量化投影）
        Q1 = self.q1_proj(f1_down)  # (B, C, h_down, w_down)
        K1 = self.k1_proj(f2_down)
        V1 = self.v1_proj(f2_down)

        # 特征图 → 序列 → MHLA输入格式
        q1_seq = Q1.flatten(2).transpose(1, 2).reshape(B, N_pieces, window_size, C)
        k1_seq = K1.flatten(2).transpose(1, 2).reshape(B, N_pieces, window_size, C)
        v1_seq = V1.flatten(2).transpose(1, 2).reshape(B, N_pieces, window_size, C)

        mhla1_input = torch.cat([q1_seq, k1_seq, v1_seq], dim=-1)  # (B, hw_down, 3C) → 适配MHLA的to_qkv
        mhla1_out = self.mhla1(mhla1_input, diff_guide1_map)  # 仅取Q通道对应的输出
        mhla1_out = mhla1_out.reshape(B, hw_down, C).transpose(1, 2).view(B, C, h_down, w_down)

        attn_weight_f1 = self.out1_proj(mhla1_out) + self.residual_scale1 * res1

        # -------------------------- 4. 方向2：f1增强f2的MHLA注意力（对称逻辑） --------------------------
        Q2 = self.q2_proj(f2_down)
        K2 = self.k2_proj(f1_down)
        V2 = self.v2_proj(f1_down)

        q2_seq = Q2.flatten(2).transpose(1, 2).reshape(B, N_pieces, window_size, C)
        k2_seq = K2.flatten(2).transpose(1, 2).reshape(B, N_pieces, window_size, C)
        v2_seq = V2.flatten(2).transpose(1, 2).reshape(B, N_pieces, window_size, C)

        mhla2_input = torch.cat([q2_seq, k2_seq, v2_seq], dim=-1)
        mhla2_out = self.mhla2(mhla2_input, diff_guide2_map)
        mhla2_out = mhla2_out.reshape(B, hw_down, C).transpose(1, 2).view(B, C, h_down, w_down)

        attn_weight_f2 = self.out2_proj(mhla2_out) + self.residual_scale2 * res2

        return attn_weight_f1, attn_weight_f2




class SequenceReduction_DifferenceFeatureComplementaryAttention(nn.Module):
    def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8, sr_ratio=1):
        super().__init__()
        self.heads = heads
        self.dim_per_head = nf // heads
        self.nf = nf
        self.threshold_coeff = threshold_coeff
        self.sr_ratio = sr_ratio  # 序列缩减比例，核心替换原downsample_ratios
        assert nf % heads == 0, "nf必须能被heads整除"
        assert isinstance(sr_ratio, int) and sr_ratio >= 1, "sr_ratio必须为≥1的正整数"

        # -------------------------- 轻量化投影层（保留原分组卷积，保证轻量化） --------------------------
        def conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
            groups = min(16, in_channels)  # 分组卷积轻量化，避免通道数过小时分组无效
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                          groups=groups, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 双向Q/K/V投影分支（2D卷积投影，后续展平为序列，保留原轻量化设计）
        self.q1_proj, self.k1_proj, self.v1_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)
        self.q2_proj, self.k2_proj, self.v2_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)

        # -------------------------- 序列缩减（SR）模块：对齐参考Attention的核心设计 --------------------------
        # 双向独立SR（f1/f2的K/V分别做缩减，保证双向注意力的独立性）
        if self.sr_ratio > 1:
            self.sr1 = nn.Conv2d(nf, nf, kernel_size=sr_ratio, stride=sr_ratio, bias=False)  # f2→f1的K/V缩减
            self.sr2 = nn.Conv2d(nf, nf, kernel_size=sr_ratio, stride=sr_ratio, bias=False)  # f1→f2的K/V缩减
            self.norm1 = nn.LayerNorm(nf)  # 序列维度归一化，对齐参考
            self.norm2 = nn.LayerNorm(nf)

        # -------------------------- 差异引导分支（仅替换下采样方式，核心逻辑不变） --------------------------
        self.diff_proj1 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)  # (B, heads, H, W)
        )
        self.diff_proj2 = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, heads, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        # -------------------------- 输出投影与残差（保留原逻辑） --------------------------
        self.out1_proj, self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False), nn.Conv2d(nf, nf, 1, bias=False)
        self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
        self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

        # 统一权重初始化：对齐参考Attention的初始化方式，保证训练稳定性
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """对齐参考Attention的权重初始化，统一卷积/线性/LayerNorm初始化"""
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, f1, f2, diff_feat):
        """
        输入：
            f1/f2: 双时相特征 (B, C, H, W)
            diff_feat: 差异特征 (B, C, H, W)
        输出：
            attn_weight_f1/f2: 双向注意力增强权重 (B, C, H, W)
        核心修改：K/V做序列缩减，Q保留原始长度，无需下采样后再上采样，计算更高效
        """
        B, C, H, W = f1.shape
        N = H * W  # 原始序列长度：H*W
        res1, res2 = f1, f2  # 残差保留原始特征

        # -------------------------- 1. 序列缩减：先对K/V特征做下采样 --------------------------
        # f2（K1/V1的输入）做序列缩减
        if self.sr_ratio > 1:
            f2_down = self.sr1(f2)  # (B, C, H/sr, W/sr)
            f1_down = self.sr2(f1)  # (B, C, H/sr, W/sr)
            # 差异特征同步下采样（和K/V尺度匹配）
            diff_feat_sr = self.sr1(diff_feat)
            h_sr, w_sr = f2_down.shape[2], f2_down.shape[3]
            N_sr = h_sr * w_sr
        else:
            f2_down = f2
            f1_down = f1
            diff_feat_sr = diff_feat
            h_sr, w_sr = H, W
            N_sr = N

        # -------------------------- 2. 差异引导权重计算（基于下采样后的差异特征） --------------------------
        # 方向1：f2增强f1的差异引导（展平为序列做计算，对齐参考）
        diff_proj1 = self.diff_proj1(diff_feat_sr).view(B, self.heads, N_sr).unsqueeze(2)  # (B, heads, 1, N_sr)
        batch_mean1 = diff_proj1.mean(dim=(-1, -2), keepdim=True)  # 动态阈值基于缩减后特征
        dynamic_threshold1 = batch_mean1 * self.threshold_coeff
        diff_guide1 = F.relu(diff_proj1 - dynamic_threshold1) + 1e-5
        diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)

        # 方向2：f1增强f2的差异引导（对称逻辑）
        diff_proj2 = self.diff_proj2(diff_feat_sr).view(B, self.heads, N_sr).unsqueeze(2)
        batch_mean2 = diff_proj2.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold2 = batch_mean2 * self.threshold_coeff
        diff_guide2 = F.relu(diff_proj2 - dynamic_threshold2) + 1e-5
        diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)

        # -------------------------- 3. Q投影：保留原始序列长度（B, heads, N, dim_per_head） --------------------------
        # 方向1：f1作为Q，2D卷积投影后展平为序列，再重塑为多头格式
        Q1 = self.q1_proj(f1).view(B, self.heads, self.dim_per_head, N).transpose(-1, -2)  # (B, heads, N, dim_per_head)
        # 方向2：f2作为Q，对称逻辑
        Q2 = self.q2_proj(f2).view(B, self.heads, self.dim_per_head, N).transpose(-1, -2)

        # -------------------------- 4. K/V投影：基于序列缩减后的特征（核心修复点） --------------------------
        # 方向1：f2_down作为K1/V1的输入（先下采样再投影，保证维度匹配）
        K1 = self.k1_proj(f2_down).view(B, self.heads, self.dim_per_head, N_sr)  # (B, heads, dim_per_head, N_sr)
        V1 = self.v1_proj(f2_down).view(B, self.heads, self.dim_per_head, N_sr)

        # 方向2：f1_down作为K2/V2的输入（对称逻辑）
        K2 = self.k2_proj(f1_down).view(B, self.heads, self.dim_per_head, N_sr)
        V2 = self.v2_proj(f1_down).view(B, self.heads, self.dim_per_head, N_sr)

        # -------------------------- 5. 注意力计算：动态缩放+差异引导 --------------------------
        # 动态缩放因子：结合缩减后序列长度，避免注意力分数过大/过小
        scale_factor = torch.sqrt(torch.tensor(N_sr, dtype=torch.float32, device=f1.device))
        dynamic_scale = (self.dim_per_head ** -0.5) * (scale_factor ** 0.5)

        # 方向1：f2增强f1的注意力计算
        score1 = torch.matmul(Q1, K1) * dynamic_scale  # (B, heads, N, N_sr)
        score1 = score1 * diff_guide1  # 应用差异引导，突出变化区域
        weight1 = self.softmax(score1)
        out1 = torch.matmul(weight1, V1.transpose(-1, -2))  # (B, heads, N, dim_per_head)
        # 重塑为2D特征图（恢复原始尺寸）
        out1 = out1.transpose(-1, -2).contiguous().view(B, C, H, W)

        # 方向2：f1增强f2的注意力计算（对称逻辑）
        score2 = torch.matmul(Q2, K2) * dynamic_scale
        score2 = score2 * diff_guide2
        weight2 = self.softmax(score2)
        out2 = torch.matmul(weight2, V2.transpose(-1, -2))
        out2 = out2.transpose(-1, -2).contiguous().view(B, C, H, W)

        # -------------------------- 6. 输出投影+残差连接 --------------------------
        attn_weight_f1 = self.out1_proj(out1) + self.residual_scale1 * res1
        attn_weight_f2 = self.out2_proj(out2) + self.residual_scale2 * res2

        return attn_weight_f1, attn_weight_f2

class Fast_SRDifferenceCrossAttention(nn.Module):
    def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8, sr_ratio=1):
        super().__init__()
        self.heads = heads
        self.dim_per_head = nf // heads
        self.nf = nf
        self.threshold_coeff = threshold_coeff
        self.sr_ratio = sr_ratio
        assert nf % heads == 0, "nf必须能被heads整除"
        assert sr_ratio >= 1, "sr_ratio≥1"

        # -------------------------- 1. 轻量化投影：单层1×1卷积替代双层分组卷积（核心提速） --------------------------
        # 移除冗余的分组卷积，直接用1×1卷积+BN+ReLU，计算量减少50%
        self.q1_proj = nn.Sequential(nn.Conv2d(nf, nf, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True))
        self.k1_proj = nn.Sequential(nn.Conv2d(nf, nf, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True))
        self.v1_proj = nn.Sequential(nn.Conv2d(nf, nf, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True))
        self.q2_proj = nn.Sequential(nn.Conv2d(nf, nf, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True))
        self.k2_proj = nn.Sequential(nn.Conv2d(nf, nf, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True))
        self.v2_proj = nn.Sequential(nn.Conv2d(nf, nf, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True))

        # -------------------------- 2. 序列缩减：共享SR卷积（双向复用，减少参数） --------------------------
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(nf, nf, kernel_size=sr_ratio, stride=sr_ratio, bias=False)  # 双向共享SR
            self.norm = nn.LayerNorm(nf)

        # -------------------------- 3. 差异引导分支：通道数减半（进一步轻量化） --------------------------
        self.diff_proj1 = nn.Sequential(
            nn.Conv2d(nf, nf // (reduction*2), 1, bias=False),  # 通道数从nf/4→nf/8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // (reduction*2), heads, 1, bias=False)
        )
        self.diff_proj2 = nn.Sequential(
            nn.Conv2d(nf, nf // (reduction*2), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // (reduction*2), heads, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        # -------------------------- 4. 输出投影：简化为单层卷积（无激活） --------------------------
        self.out1_proj = nn.Conv2d(nf, nf, 1, bias=False)
        self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False)
        self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
        self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    # ✅ 修复：将JIT函数改为静态方法，避免self参数干扰
    @staticmethod
    @torch.jit.script
    def fast_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                       diff_guide: torch.Tensor, dynamic_scale: torch.Tensor) -> torch.Tensor:
        """独立封装注意力计算，静态方法+JIT编译，用F.softmax替代nn.Softmax"""
        score = torch.matmul(Q, K) * dynamic_scale
        score = score * diff_guide
        weight = F.softmax(score, dim=-1)  # 改用functional版softmax，JIT可识别
        out = torch.matmul(weight, V.transpose(-1, -2))
        return out

    def forward(self, f1, f2, diff_feat):
        B, C, H, W = f1.shape
        N = H * W
        res1, res2 = f1, f2

        # -------------------------- 1. 序列缩减：共享SR，减少计算 --------------------------
        if self.sr_ratio > 1:
            f2_down = self.sr(f2)
            f1_down = self.sr(f1)
            diff_feat_sr = self.sr(diff_feat)
            h_sr, w_sr = f2_down.shape[2], f2_down.shape[3]
            N_sr = h_sr * w_sr
        else:
            f2_down = f2
            f1_down = f1
            diff_feat_sr = diff_feat
            h_sr, w_sr = H, W
            N_sr = N

        # -------------------------- 2. 差异引导：计算逻辑不变，通道数更少 --------------------------
        diff_proj1 = self.diff_proj1(diff_feat_sr).view(B, self.heads, N_sr).unsqueeze(2)
        batch_mean1 = diff_proj1.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold1 = batch_mean1 * self.threshold_coeff
        diff_guide1 = F.relu(diff_proj1 - dynamic_threshold1) + 1e-5
        diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)

        diff_proj2 = self.diff_proj2(diff_feat_sr).view(B, self.heads, N_sr).unsqueeze(2)
        batch_mean2 = diff_proj2.mean(dim=(-1, -2), keepdim=True)
        dynamic_threshold2 = batch_mean2 * self.threshold_coeff
        diff_guide2 = F.relu(diff_proj2 - dynamic_threshold2) + 1e-5
        diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)

        # -------------------------- 3. Q/K/V投影：简化后计算更快 --------------------------
        # Q投影（原始尺寸）
        Q1 = self.q1_proj(f1).view(B, self.heads, self.dim_per_head, N).transpose(-1, -2)
        Q2 = self.q2_proj(f2).view(B, self.heads, self.dim_per_head, N).transpose(-1, -2)

        # K/V投影（下采样尺寸）
        K1 = self.k1_proj(f2_down).view(B, self.heads, self.dim_per_head, N_sr)
        V1 = self.v1_proj(f2_down).view(B, self.heads, self.dim_per_head, N_sr)
        K2 = self.k2_proj(f1_down).view(B, self.heads, self.dim_per_head, N_sr)
        V2 = self.v2_proj(f1_down).view(B, self.heads, self.dim_per_head, N_sr)

        # -------------------------- 4. 注意力计算：JIT编译加速 --------------------------
        dynamic_scale = (self.dim_per_head ** -0.5) * math.sqrt(N_sr)  # 提前计算常数，减少张量操作
        dynamic_scale = torch.tensor(dynamic_scale, device=f1.device, dtype=torch.float32)

        # 调用JIT编译的注意力函数
        out1 = self.fast_attention(Q1, K1, V1, diff_guide1, dynamic_scale)
        out1 = out1.transpose(-1, -2).contiguous().view(B, C, H, W)

        out2 = self.fast_attention(Q2, K2, V2, diff_guide2, dynamic_scale)
        out2 = out2.transpose(-1, -2).contiguous().view(B, C, H, W)

        # -------------------------- 5. 输出：简化投影 --------------------------
        attn_weight_f1 = self.out1_proj(out1) + self.residual_scale1 * res1
        attn_weight_f2 = self.out2_proj(out2) + self.residual_scale2 * res2

        return attn_weight_f1, attn_weight_f2


# 测试代码：适配你的4个stage特征尺寸
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Stage1: 64×64, sr_ratio=8; Stage2:32×32,sr_ratio=4; Stage3:16×16,sr_ratio=2; Stage4:8×8,sr_ratio=1
    attn_stage1 = SequenceReduction_DifferenceFeatureComplementaryAttention(nf=64, heads=2, sr_ratio=8).to(device)
    attn_stage2 = SequenceReduction_DifferenceFeatureComplementaryAttention(nf=128, heads=2, sr_ratio=4).to(device)
    attn_stage3 = SequenceReduction_DifferenceFeatureComplementaryAttention(nf=256, heads=4, sr_ratio=2).to(device)
    attn_stage4 = SequenceReduction_DifferenceFeatureComplementaryAttention(nf=512, heads=8, sr_ratio=1).to(device)

    # 测试Stage1特征
    f1 = torch.randn(16, 64, 64, 64).to(device)
    f2 = torch.randn(16, 64, 64, 64).to(device)
    diff = torch.randn(16, 64, 64, 64).to(device)
    out1, out2 = attn_stage1(f1, f2, diff)
    print(f"Stage1输出形状: {out1.shape}, {out2.shape}")  # 输出：torch.Size([16, 64, 64, 64]), torch.Size([16, 64, 64, 64])

    # 测试Stage4（无序列缩减）
    f1_4 = torch.randn(16, 512, 8, 8).to(device)
    f2_4 = torch.randn(16, 512, 8, 8).to(device)
    diff_4 = torch.randn(16, 512, 8, 8).to(device)
    out1_4, out2_4 = attn_stage4(f1_4, f2_4, diff_4)
    print(f"Stage4输出形状: {out1_4.shape}, {out2_4.shape}")  # 输出：torch.Size([16, 512, 8, 8]), torch.Size([16, 512, 8, 8])