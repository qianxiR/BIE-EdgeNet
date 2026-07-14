import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from models import CBAM
from models.submodules import *

from torchvision.models import ResNet34_Weights


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# self attention--spatial
class Atten_Spa(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Spa, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()

        query = self.query_conv(x)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


# cross attention--spatial w/ change guide map
class Atten_Cross(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Cross, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, cond):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()

        # guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True) # map 2,1,128,128 -> 2,1,16,16
        #
        # guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) # query from x
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(cond)  # key from cond
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(cond) # value from cond
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

# class LightWeight_DifferenceFeatureComplementaryAttention(nn.Module):
#     def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8):
#         super().__init__()
#         self.heads = heads
#         self.dim_per_head = nf // heads
#         self.nf = nf
#         self.threshold_coeff = threshold_coeff
#         assert nf % heads == 0, "nf必须能被heads整除"

#         # -------------------------- 轻量化投影层（保留分组卷积，降低计算量） --------------------------
#         def conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
#             groups = min(16, in_channels)  # 分组卷积轻量化
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
#                           groups=groups, bias=False),
#                 nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )

#         # 双向投影分支（输出通道仍为nf，分注意力头）
#         self.q1_proj, self.k1_proj, self.v1_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)
#         self.q2_proj, self.k2_proj, self.v2_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)

#         # -------------------------- 差异引导分支（输出通道=heads，与原逻辑一致） --------------------------
#         self.diff_proj1 = nn.Sequential(
#             nn.Conv2d(nf, nf // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf // reduction, heads, 1, bias=False)  # (B, heads, H, W)
#         )
#         self.diff_proj2 = nn.Sequential(
#             nn.Conv2d(nf, nf // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf // reduction, heads, 1, bias=False)
#         )

#         self.softmax = nn.Softmax(dim=-1)

#         # -------------------------- 输出投影与残差（保留原逻辑） --------------------------
#         self.out1_proj, self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False), nn.Conv2d(nf, nf, 1, bias=False)
#         self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
#         self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

#         # -------------------------- 新增：动态下采样比例映射（适配多尺度输入） --------------------------
#         # 输入空间尺寸 → 下采样比例（小尺寸特征不下采样，避免信息丢失）
#         self.downsample_ratios = {
#             128: 4,
#             64: 4,   # 64×64 → 16×16（4倍下采样）
#             32: 4,   # 32×32 → 8×8（4倍下采样）
#             16: 2,   # 16×16 → 8×8（2倍下采样）
#             8: 1     # 8×8 → 8×8（不下采样）
#         }

#     def forward(self, f1, f2, diff_feat):
#         B, C, H, W = f1.shape  # 输入尺寸：(1, 64, 64, 64) → H=64, W=64；以此类推
#         res1, res2 = f1, f2    # 残差保留原始特征

#         # -------------------------- 1. 动态下采样（核心优化：根据输入尺寸调整） --------------------------
#         # 获取当前特征的下采样比例（默认1，即不下采样）
#         down_ratio = self.downsample_ratios.get(H, 1)
#         if down_ratio > 1:
#             # 下采样特征（使用平均池化，保留全局信息）
#             f1_down = F.avg_pool2d(f1, kernel_size=down_ratio, stride=down_ratio)  # (B, C, H/ratio, W/ratio)
#             f2_down = F.avg_pool2d(f2, kernel_size=down_ratio, stride=down_ratio)
#             diff_feat_down = F.avg_pool2d(diff_feat, kernel_size=down_ratio, stride=down_ratio)
#             h_down, w_down = f1_down.shape[2], f1_down.shape[3]  # 下采样后的尺寸
#             hw_down = h_down * w_down  # 下采样后的空间元素数
#         else:
#             # 小尺寸特征（如8×8）不下采样，直接使用原始特征
#             f1_down, f2_down, diff_feat_down = f1, f2, diff_feat
#             h_down, w_down = H, W
#             hw_down = H * W

#         # -------------------------- 2. 差异引导权重计算（基于下采样特征，显存友好） --------------------------
#         # 方向1：f2增强f1的差异引导
#         # 下采样特征投影 → 展平为 (B, heads, hw_down) → 增加维度用于广播
#         diff_proj1 = self.diff_proj1(diff_feat_down).view(B, self.heads, hw_down).unsqueeze(2)  # (B, heads, 1, hw_down)
#         # 动态阈值（基于下采样后的特征均值）
#         batch_mean1 = diff_proj1.mean(dim=(-1, -2), keepdim=True)  # (B, heads, 1, 1)
#         dynamic_threshold1 = batch_mean1 * self.threshold_coeff
#         # 过滤弱关联并归一化
#         diff_guide1 = F.relu(diff_proj1 - dynamic_threshold1) + 1e-5
#         diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)

#         # 方向2：f1增强f2的差异引导（对称逻辑）
#         diff_proj2 = self.diff_proj2(diff_feat_down).view(B, self.heads, hw_down).unsqueeze(2)
#         batch_mean2 = diff_proj2.mean(dim=(-1, -2), keepdim=True)
#         dynamic_threshold2 = batch_mean2 * self.threshold_coeff
#         diff_guide2 = F.relu(diff_proj2 - dynamic_threshold2) + 1e-5
#         diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)

#         # -------------------------- 3. Q/K/V投影与注意力计算（下采样特征上操作） --------------------------
#         # 方向1：f2增强f1的注意力
#         # Q1: 下采样特征投影 → 重塑为 (B, heads, hw_down, dim_per_head)
#         Q1 = self.q1_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)  # (B, heads, hw_down, dim)
#         # K1: 下采样特征投影 → (B, heads, dim_per_head, hw_down)
#         K1 = self.k1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)
#         # V1: 下采样特征投影 → (B, heads, dim_per_head, hw_down)
#         V1 = self.v1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)

#         # 注意力分数计算（基于下采样尺寸，矩阵尺寸大幅减小）
#         scale_factor = torch.sqrt(torch.tensor(hw_down, dtype=torch.float32, device=f1.device))
#         dynamic_scale = (self.dim_per_head ** -0.5) * (scale_factor ** 0.5)
#         score1 = torch.matmul(Q1, K1) * dynamic_scale  # (B, heads, hw_down, hw_down)
#         score1 = score1 * diff_guide1  # 应用差异引导

#         # 注意力加权与上采样恢复
#         weight1 = self.softmax(score1)  # (B, heads, hw_down, hw_down)
#         out1 = torch.matmul(V1, weight1.transpose(-1, -2))  # (B, heads, dim_per_head, hw_down)
#         out1 = out1.view(B, self.heads * self.dim_per_head, h_down, w_down)  # 重塑为特征图
#         # 上采样回原始尺寸（与输入f1同形）
#         out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=True)
#         # 输出投影 + 残差连接
#         attn_weight_f1 = self.out1_proj(out1) + self.residual_scale1 * res1

#         # -------------------------- 4. 方向2：f1增强f2的注意力（对称逻辑） --------------------------
#         Q2 = self.q2_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)
#         K2 = self.k2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)
#         V2 = self.v2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)

#         score2 = torch.matmul(Q2, K2) * dynamic_scale
#         score2 = score2 * diff_guide2

#         weight2 = self.softmax(score2)
#         out2 = torch.matmul(V2, weight2.transpose(-1, -2)).view(B, C, h_down, w_down)
#         out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=True)  # 上采样恢复
#         attn_weight_f2 = self.out2_proj(out2) + self.residual_scale2 * res2

#         return attn_weight_f1, attn_weight_f2

# class DW_DifferenceFeatureComplementaryAttention(nn.Module):
#     def __init__(self, nf, heads=2, reduction=4, threshold_coeff=0.8):
#         super().__init__()
#         self.heads = heads
#         self.dim_per_head = nf // heads
#         self.nf = nf
#         self.threshold_coeff = threshold_coeff
#         assert nf % heads == 0, "nf必须能被heads整除"

#         # -------------------------- 改进1：深度可分离卷积替换分组卷积（更轻量化） --------------------------
#         def conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
#             # 深度可分离卷积：depthwise + pointwise，计算量降低≈groups倍
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
#                           groups=in_channels, bias=False),  # depthwise
#                 nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),  # pointwise
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.1)  # 改进2：加入Dropout提升泛化性
#             )

#         # 双向投影分支
#         self.q1_proj, self.k1_proj, self.v1_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)
#         self.q2_proj, self.k2_proj, self.v2_proj = conv_bn(nf, nf), conv_bn(nf, nf), conv_bn(nf, nf)

#         # -------------------------- 差异引导分支（新增局部上下文增强） --------------------------
#         self.diff_proj1 = nn.Sequential(
#             nn.Conv2d(nf, nf // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf // reduction, heads, 3, padding=1, groups=heads, bias=False),  # 局部卷积增强空间连续性
#             nn.BatchNorm2d(heads)
#         )
#         self.diff_proj2 = nn.Sequential(
#             nn.Conv2d(nf, nf // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf // reduction, heads, 3, padding=1, groups=heads, bias=False),
#             nn.BatchNorm2d(heads)
#         )
#         self.diff_context = nn.Conv2d(heads, heads, 3, padding=1, groups=heads, bias=False)  # 差异上下文增强

#         self.softmax = nn.Softmax(dim=-1)
#         self.attn_drop = nn.Dropout(0.1)  # 改进3：注意力权重Dropout

#         # -------------------------- 输出投影与残差（自适应残差缩放） --------------------------
#         self.out1_proj, self.out2_proj = nn.Conv2d(nf, nf, 1, bias=False), nn.Conv2d(nf, nf, 1, bias=False)
#         self.residual_scale1 = nn.Parameter(torch.tensor(0.1))
#         self.residual_scale2 = nn.Parameter(torch.tensor(0.1))

#         # -------------------------- 改进4：自适应下采样（替代硬编码字典） --------------------------
#         self.min_down_size = 8  # 下采样后最小尺寸

#         # 改进5：可学习温度系数（替代静态scale）
#         self.temp = nn.Parameter(torch.tensor(1.0))

#         # 改进6：多头动态融合权重
#         self.head_weights = nn.Parameter(torch.ones(self.heads))

#     def get_adaptive_down_ratio(self, H):
#         """根据输入尺寸自适应计算下采样比例"""
#         base_ratio = 4
#         ratio = min(base_ratio, H // self.min_down_size)
#         return max(1, ratio)

#     def forward(self, f1, f2, diff_feat):
#         B, C, H, W = f1.shape
#         res1, res2 = f1, f2

#         # 1. 自适应下采样
#         down_ratio = self.get_adaptive_down_ratio(H)
#         if down_ratio > 1:
#             f1_down = F.avg_pool2d(f1, kernel_size=down_ratio, stride=down_ratio)
#             f2_down = F.avg_pool2d(f2, kernel_size=down_ratio, stride=down_ratio)
#             diff_feat_down = F.avg_pool2d(diff_feat, kernel_size=down_ratio, stride=down_ratio)
#             h_down, w_down = f1_down.shape[2], f1_down.shape[3]
#             hw_down = h_down * w_down
#         else:
#             f1_down, f2_down, diff_feat_down = f1, f2, diff_feat
#             h_down, w_down = H, W
#             hw_down = H * W

#         # 2. 差异引导权重计算（改进：分位数阈值+上下文增强）
#         # 方向1：f2增强f1的差异引导
#         diff_proj1 = self.diff_proj1(diff_feat_down)  # (B, heads, h_down, w_down)
#         diff_proj1 = self.diff_context(diff_proj1)  # 局部上下文增强
#         diff_proj1 = diff_proj1.view(B, self.heads, hw_down).unsqueeze(2)  # (B, heads, 1, hw_down)

#         # 改进：分位数阈值（抗噪声，替代均值阈值）
#         quantile_90 = torch.quantile(diff_proj1, 0.9, dim=-1, keepdim=True)
#         dynamic_threshold1 = quantile_90 * self.threshold_coeff

#         # 区分正负差异
#         diff_pos = F.relu(diff_proj1 - dynamic_threshold1)
#         diff_neg = F.relu(dynamic_threshold1 - diff_proj1)
#         diff_alpha = nn.Parameter(torch.tensor(0.7, device=f1.device))  # 可学习正差异权重
#         diff_guide1 = diff_alpha * diff_pos + (1 - diff_alpha) * diff_neg
#         diff_guide1 = diff_guide1 / (diff_guide1.sum(dim=-1, keepdim=True) + 1e-6)

#         # 方向2：f1增强f2的差异引导（对称逻辑）
#         diff_proj2 = self.diff_proj2(diff_feat_down)
#         diff_proj2 = self.diff_context(diff_proj2)
#         diff_proj2 = diff_proj2.view(B, self.heads, hw_down).unsqueeze(2)

#         quantile_90_2 = torch.quantile(diff_proj2, 0.9, dim=-1, keepdim=True)
#         dynamic_threshold2 = quantile_90_2 * self.threshold_coeff

#         diff_pos2 = F.relu(diff_proj2 - dynamic_threshold2)
#         diff_neg2 = F.relu(dynamic_threshold2 - diff_proj2)
#         diff_guide2 = diff_alpha * diff_pos2 + (1 - diff_alpha) * diff_neg2
#         diff_guide2 = diff_guide2 / (diff_guide2.sum(dim=-1, keepdim=True) + 1e-6)

#         # 3. Q/K/V投影与注意力计算
#         # 方向1：f2增强f1的注意力
#         Q1 = self.q1_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)
#         K1 = self.k1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)
#         V1 = self.v1_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down)

#         # 改进：可学习温度系数的动态缩放
#         scale_factor = torch.sqrt(torch.tensor(hw_down, dtype=torch.float32, device=f1.device))
#         dynamic_scale = (self.dim_per_head ** -0.5) * (scale_factor ** 0.5) * self.temp
#         score1 = torch.matmul(Q1, K1) * dynamic_scale
#         score1 = score1 * diff_guide1

#         # 注意力加权（加入Dropout）
#         weight1 = self.softmax(score1)
#         weight1 = self.attn_drop(weight1)  # 正则化注意力权重

#         out1 = torch.matmul(V1, weight1.transpose(-1, -2))
#         # 改进：多头动态融合
#         out1 = out1 * self.head_weights.view(1, self.heads, 1, 1)
#         out1 = out1.view(B, self.heads * self.dim_per_head, h_down, w_down)

#         # 改进：可学习上采样（替代双线性插值）
#         if down_ratio > 1:
#             out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=True)
#         # 自适应残差缩放（基于特征方差）
#         res1_var = torch.var(res1, dim=(2, 3), keepdim=True)
#         attn_weight_f1 = self.out1_proj(out1) + (self.residual_scale1 * res1_var) * res1

#         # 方向2：f1增强f2的注意力（对称逻辑）
#         Q2 = self.q2_proj(f2_down).view(B, self.heads, self.dim_per_head, hw_down).transpose(-1, -2)
#         K2 = self.k2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)
#         V2 = self.v2_proj(f1_down).view(B, self.heads, self.dim_per_head, hw_down)

#         score2 = torch.matmul(Q2, K2) * dynamic_scale
#         score2 = score2 * diff_guide2

#         weight2 = self.softmax(score2)
#         weight2 = self.attn_drop(weight2)

#         out2 = torch.matmul(V2, weight2.transpose(-1, -2))
#         out2 = out2 * self.head_weights.view(1, self.heads, 1, 1)
#         out2 = out2.view(B, C, h_down, w_down)

#         if down_ratio > 1:
#             out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=True)
#         res2_var = torch.var(res2, dim=(2, 3), keepdim=True)
#         attn_weight_f2 = self.out2_proj(out2) + (self.residual_scale2 * res2_var) * res2

#         return attn_weight_f1, attn_weight_f2

# class Pre_Post_TemporalBIE(nn.Module):
#     """双向跨时相BIE模块：输出两个增强特征（f2增强f1 + f1增强f2）"""

#     def __init__(self, nf, heads, reduction):
#         super().__init__()
#         # 特征对齐：统一x1/x2特征分布（确保双时相特征可对比）
#         self.align1 = nn.Conv2d(nf, nf, 1, bias=False)
#         self.align2 = nn.Conv2d(nf, nf, 1, bias=False)

#         # # 双时相注意力：基于f1、f2和差异生成权重（覆盖双向视角）
#         # self.attn = nn.Sequential(
#         #     nn.Conv2d(nf * 3, nf, 1),  # 输入：f1 + f2 + 差异特征（3*nf通道）
#         #     nn.ReLU(inplace=False),
#         #     nn.Conv2d(nf, nf, 1),
#         #     nn.Sigmoid()  # 注意力权重（0~1，突出变化区域）
#         # )
#         # 替换为轻量化双向交叉注意力（核心优化）
#         self.attn = LightWeight_DifferenceFeatureComplementaryAttention(nf, heads=heads, reduction=reduction)
#         # self.attn = DW_DifferenceFeatureComplementaryAttention(nf, heads=heads, reduction=reduction)

#         self.CBAM1 = CBAM.CBAMBlock(nf)
#         self.CBAM2 = CBAM.CBAMBlock(nf)
#         # self.scsa1 = SCSA(nf, head_num=16)
#         # self.scsa2 = SCSA(nf, head_num=16)

#         # 单独归一化：分别稳定两个增强特征的分布
#         self.norm1 = LayerNorm2d(nf)  # 用于enhanced_f1
#         self.norm2 = LayerNorm2d(nf)  # 用于enhanced_f2

#         # 权重初始化
#         nn.init.normal_(self.align1.weight, mean=0, std=0.01)
#         nn.init.normal_(self.align2.weight, mean=0, std=0.01)
#         initialize_weights(self.attn, 0.1)
#         self.sigmoid = nn.Sigmoid()


#         # self.SMSA1 = Shareable_Multi_Semantic_Spatial_Attention(nf)
#         # self.SMSA2 = Shareable_Multi_Semantic_Spatial_Attention(nf)

#     # def spatial_difference(self, xA, xB):
#     #     xA_flat = xA.permute(0, 2, 3, 1).reshape(-1, xA.size(1))
#     #     xB_flat = xB.permute(0, 2, 3, 1).reshape(-1, xB.size(1))
#     #     cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)
#     #     cosine_sim = cosine_sim.view(xA.size(0), xA.size(2), xA.size(3))
#     #     cosine_sim = cosine_sim.unsqueeze(1)
#     #     c_weights = 1-self.sigmoid(cosine_sim)
#     #     return c_weights
#     def spatial_difference(self, xA, xB):
#         B, C, H, W = xA.shape
#         # 保留空间维度：(B, C, H, W) → (B, C, H*W)，每个位置对应一个特征向量
#         xA_flat = xA.view(B, C, H*W)  # (B, C, HW)
#         xB_flat = xB.view(B, C, H*W)  # (B, C, HW)
#         # 按通道维度计算每个空间位置的余弦相似度（dim=1）
#         cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)  # (B, HW)
#         # 恢复空间维度：(B, 1, H, W)，与输入特征同尺寸
#         cosine_sim = cosine_sim.view(B, 1, H, W)
#         c_weights = 1 - self.sigmoid(cosine_sim)  # 差异越大，权重越高
#         return c_weights

#     def channel_difference(self, xA, xB):
#         N, C, H, W = xA.shape
#         xA_flat = xA.view(N, C, -1)
#         xB_flat = xB.view(N, C, -1)
#         cosine_sim = 1-self.sigmoid(F.cosine_similarity(xA_flat, xB_flat, dim=2))
#         hw_weights = cosine_sim.unsqueeze(-1).unsqueeze(-1)
#         return hw_weights

#     def forward(self, feat_x1, feat_x2):
#         """
#         Args:
#             feat_x1: x1单时相特征 [B, C, H, W]
#             feat_x2: x2单时相特征 [B, C, H, W]
#         Returns:
#             enhanced_f1: 用f2增强后的x1特征 [B, C, H, W]
#             enhanced_f2: 用f1增强后的x2特征 [B, C, H, W]
#         """
#         # 1. 特征对齐：消除双时相特征分布差异
#         f1 = self.align1(feat_x1)  # x1特征对齐
#         f2 = self.align2(feat_x2)  # x2特征对齐

#         # 计算空间权重和通道权重
#         c_weights = self.spatial_difference(f1, f2)  # (2, 1, 32, 32)
#         hw_weights = self.channel_difference(f1, f2)  # (2, 128, 1, 1)

#         # 将 c_weights 扩展到与 hw_weights 相同的形状
#         c_weights_expanded = c_weights.expand(-1, hw_weights.size(1), -1, -1)  # (2, 128, 32, 32)

#         # 合并权重 (比如可以选择相乘，也可以进行加权平均)
#         combined_weights = c_weights_expanded * hw_weights  # (2, 128, 32, 32)

#         # # 对 xA 和 xB 进行加权处理
#         # xA_weighted = f1 * combined_weights
#         # xB_weighted = f2 * combined_weights
#         #
#         # # 2. 差异特征与双向注意力：捕捉双时相变化
#         abs_diff = torch.abs(f1 - f2)  # 显式计算变化区域
#         diff_feat = abs_diff + abs_diff * combined_weights

#         # attn_weight = self.attn(torch.cat([f1, f2, diff_feat], dim=1))  # 融合双时相视角生成权重
#         #
#         # # 3. 双向增强：用对方特征强化自身变化区域
#         # # enhanced_f1：x1的基础上，用x2的变化区域特征增强
#         # enhanced_f1 = f1 + attn_weight * f2
#         # # enhanced_f2：x2的基础上，用x1的变化区域特征增强
#         # enhanced_f2 = f2 + attn_weight * f1

#         # 4. 轻量化双向交叉注意力（输出两个方向的权重）
#         attn_weight_f1, attn_weight_f2 = self.attn(f1, f2, diff_feat)

#         # 5. 双向增强
#         enhanced_f1 = f1 + attn_weight_f1 * f2
#         enhanced_f2 = f2 + attn_weight_f2 * f1

#         # enhanced_f1 = self.CBAM1(enhanced_f1)
#         # enhanced_f2 = self.CBAM2(enhanced_f2)

#         # 4. 残差归一化：保留各自原始特征，稳定梯度
#         # 对enhanced_f1：残差连接x1的原始特征（确保x1基础信息不丢失）
#         enhanced_f1 = self.norm1(enhanced_f1 + feat_x1)
#         # 对enhanced_f2：残差连接x2的原始特征（确保x2基础信息不丢失）
#         enhanced_f2 = self.norm2(enhanced_f2 + feat_x2)

#         # if self.ifSMSA:
#         #     # enhanced_f1 = self.SMSA1(enhanced_f1)
#         #     # enhanced_f2 = self.SMSA2(enhanced_f2)
#         #     enhanced_f1 = self.scsa1(enhanced_f1)
#         #     enhanced_f2 = self.scsa2(enhanced_f2)
#         # else:
#         enhanced_f1 = self.CBAM1(enhanced_f1)
#         enhanced_f2 = self.CBAM2(enhanced_f2)


#         return enhanced_f1, enhanced_f2, diff_feat # 输出两个增强特征

#增加decoder解码器，不concat多尺度特征生成guide map,也concat多尺度特征生成最后输出
class HSANet(nn.Module):
    def __init__(self,):
        super(HSANet, self).__init__()
        # vgg16_bn = models.vgg16_bn(pretrained=False)
        # self.inc = vgg16_bn.features[:5]  # 64
        # self.down1 = vgg16_bn.features[5:12]  # 128
        # self.down2 = vgg16_bn.features[12:22]  # 256
        # self.down3 = vgg16_bn.features[22:32]  # 512
        # self.down4 = vgg16_bn.features[32:42]  # 512
        resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)  # 替代deprecated的pretrained

        
        # 基础特征提取：3→64通道（conv1+bn1+relu）
        self.resnet_base = nn.Sequential(
            resnet34.conv1,      # 3→64, 128×128
            resnet34.bn1,
            resnet34.relu
        )
        
        # 原始VGG inc: 64通道/256×256 → 基础特征+2倍上采样
        self.inc = nn.Sequential(
            self.resnet_base,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128×128→256×256
        )
        
        # 原始VGG down1: 128通道/128×128 → maxpool+layer1 + 升维融合
        self.down1_backbone = nn.Sequential(
            resnet34.maxpool,    # 64×64（输入是resnet_base的64通道特征）
            resnet34.layer1      # 64→64, 64×64
        )
        self.down1_dim = BasicConv2d(64, 128, kernel_size=1)  # 64→128通道，匹配VGG down1
        
        # 原始VGG down2: 256通道/64×64 → layer2 + 升维融合
        self.down2_backbone = resnet34.layer2  # 64→128, 32×32（输入是layer1的64通道）
        self.down2_dim = BasicConv2d(128, 256, kernel_size=1) # 128→256通道
        
        # 原始VGG down3: 512通道/32×32 → layer3 + 升维融合
        self.down3_backbone = resnet34.layer3  # 128→256, 16×16（输入是layer2的128通道）
        self.down3_dim = BasicConv2d(256, 512, kernel_size=1) # 256→512通道
        
        # 原始VGG down4: 512通道/16×16 → layer4 + 特征融合
        self.down4_backbone = resnet34.layer4  # 256→512, 8×8（输入是layer3的256通道）
        self.down4_dim = BasicConv2d(512, 512, kernel_size=1) # 保持512通道，特征融合

        # self.temporal_bie1 = Pre_Post_TemporalBIE(nf=128, heads=4, reduction=2)
        # self.temporal_bie2 = Pre_Post_TemporalBIE(nf=256, heads=2, reduction=4)
        # self.temporal_bie3 = Pre_Post_TemporalBIE(nf=512, heads=2, reduction=8)
        # self.temporal_bie4 = Pre_Post_TemporalBIE(nf=512, heads=2, reduction=8)

        self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)

        self.up_layer4 = BasicConv2d(512,512,3,1,1)
        self.up_layer3 = BasicConv2d(512,512,3,1,1)
        self.up_layer2 = BasicConv2d(256,256,3,1,1)

        self.decoder = nn.Sequential(BasicConv2d(512,64,3,1,1),
                                     nn.Conv2d(64,1,3,1,1))

        self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1),
                                           nn.Conv2d(64, 1, 1))

        self.cgm_1 = Atten_Cross(128)
        self.cgm_2 = Atten_Cross(256)
        self.cgm_3 = Atten_Cross(512)
        self.cgm_4 = Atten_Cross(512)

        self.sa_1 = Atten_Spa(128)
        self.sa_2 = Atten_Spa(256)
        self.sa_3 = Atten_Spa(512)
        self.sa_4 = Atten_Spa(512)

        #相比v2 额外的模块
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024,512,3,1,1)
        self.decoder_module3 = BasicConv2d(768,256,3,1,1)
        self.decoder_module2 = BasicConv2d(384,128,3,1,1)

    def extract_resnet_features(self, x):
        """按顺序提取ResNet34特征，保证通道匹配"""
       # Step1: 先提取基础特征（3→64通道，128×128）
        x_base = self.resnet_base(x)  # 64, 128×128
        
        # Step2: 提取inc特征（匹配VGG inc: 64/256×256）
        x_inc = self.inc[1](x_base)   # 仅用upsample，避免重复过conv1
        
        # Step3: 提取down1特征（匹配VGG down1: 128/128×128）
        x_down1_backbone = self.down1_backbone(x_base)  # 输入是64通道x_base，不是原始x！
        x_down1_dim = self.down1_dim(x_down1_backbone)  # 64→128
        x_down1 = self.upsample2x(x_down1_dim)          # 64×64→128×128
        
        # Step4: 提取down2特征（匹配VGG down2: 256/64×64）
        x_down2_backbone = self.down2_backbone(x_down1_backbone)  # 输入是layer1的64通道
        x_down2_dim = self.down2_dim(x_down2_backbone)            # 128→256
        x_down2 = self.upsample2x(x_down2_dim)                    # 32×32→64×64
        
        # Step5: 提取down3特征（匹配VGG down3: 512/32×32）
        x_down3_backbone = self.down3_backbone(x_down2_backbone)  # 输入是layer2的128通道
        x_down3_dim = self.down3_dim(x_down3_backbone)            # 256→512
        x_down3 = self.upsample2x(x_down3_dim)                    # 16×16→32×32
        
        # Step6: 提取down4特征（匹配VGG down4: 512/16×16）
        x_down4_backbone = self.down4_backbone(x_down3_backbone)  # 输入是layer3的256通道
        x_down4_dim = self.down4_dim(x_down4_backbone)            # 512→512
        x_down4 = self.upsample2x(x_down4_dim)                    # 8×8→16×16
        
        # 返回顺序：down1, down2, down3, down4（匹配原始代码的返回逻辑）
        return x_down1, x_down2, x_down3, x_down4

    def forward(self,A,B):

        size = A.size()[2:]
        # layer1_pre = self.inc(A) # 2,64,256,256
        # layer1_A = self.down1(layer1_pre) # 2,128,128,128
        # layer2_A = self.down2(layer1_A) # 2,256,64,64
        # layer3_A = self.down3(layer2_A) # 2,512,32,32
        # layer4_A = self.down4(layer3_A) # 2,512,16,16
        # 处理A分支（ResNet34层级）

        # 提取A和B的ResNet特征（保证通道/尺寸匹配原VGG）
        layer1_A, layer2_A, layer3_A, layer4_A = self.extract_resnet_features(A)
        layer1_B, layer2_B, layer3_B, layer4_B = self.extract_resnet_features(B)

        # enhanced_layer1_A, enhanced_layer1_B, diff_feat_1 = self.temporal_bie1(layer1_A, layer1_B)  # 16, 128, 128, 128
        # enhanced_layer2_A, enhanced_layer2_B, diff_feat_2 = self.temporal_bie2(layer2_A, layer2_B)  # 16, 256, 64, 64
        # enhanced_layer3_A, enhanced_layer3_B, diff_feat_3 = self.temporal_bie3(layer3_A, layer3_B)  # 16, 512, 32, 32
        # enhanced_layer4_A, enhanced_layer4_B, diff_feat_4 = self.temporal_bie4(layer4_A, layer4_B)  # 16, 512, 16, 16

        # Concatenate features from A and B
        layer1 = self.conv_reduce_1(torch.cat((layer1_B, layer1_A), dim=1)) # 2,128,128,128
        layer2 = self.conv_reduce_2(torch.cat((layer2_B, layer2_A), dim=1)) # 2,256,64,64
        layer3 = self.conv_reduce_3(torch.cat((layer3_B, layer3_A), dim=1)) # 2,512,32,32
        layer4 = self.conv_reduce_4(torch.cat((layer4_B, layer4_A), dim=1)) # 2,512,16,16

        # # change semantic guiding map 这部分没用到
        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True) # 2,512,128,128
        feature_fuse=layer4_1 #需要注释！
        change_map = self.decoder(feature_fuse)  #2,1,128,128

        # self attention
        layer4_4 = self.sa_4(layer4)
        layer4_5 = self.cgm_4(layer4_4,layer4)
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(layer4_5), layer3], 1))

        layer3_3 = self.sa_3(feature4)
        layer3_4 = self.cgm_3(layer3_3,layer3)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3_4),layer2],1))

        layer2_3 = self.sa_2(feature3)
        layer2_4 = self.cgm_2(layer2_3,layer2)
        feature2 = self.decoder_module2(torch.cat([self.upsample2x(layer2_4), layer1], 1))

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(feature2)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map


if __name__=='__main__':
    #测试热图
    net = HSANet().cuda()
    out, _ = net(torch.rand((16, 3, 256, 256)).cuda(), torch.rand((16, 3, 256, 256)).cuda())
    print(out.size())

