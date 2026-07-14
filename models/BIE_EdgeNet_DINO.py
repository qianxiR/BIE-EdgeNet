from typing import Optional, Tuple, Union, Dict
import torch.nn.functional
from functools import partial
from torchvision import models
from models.TransformerBaseNetworks import *
from models import CBAM
from models.base_model import ASPP_v1, OverlapPatchEmbed, Diff_SEM, BilinearUp, Block, \
    EdgeFusion, DiffSementicAddEdge
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_
import math
import warnings

from models.submodules import *

from models.ChangeDINO.blocks.adapter import DINOV3Wrapper, DenseAdapterLite
# torch.autograd.set_detect_anomaly(True)
from models.BIE_EdgeNet_DINO_MV2 import ResLayers_Make

from einops import rearrange

def get_config(mode: str = "xxs") -> dict:

    if mode == "x_small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 128,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 256,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 1,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },

            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 15,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 128,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 15,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 256,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 1,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },

            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    else:
        raise NotImplementedError

    for k in ["layer3"]:
        config[k].update({"dropout": 0.1, "ffn_dropout": 0.0, "attn_dropout": 0.0})

    return config

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

class Pre_Post_TemporalBIE(nn.Module):
    """双向跨时相BIE模块：输出两个增强特征（f2增强f1 + f1增强f2）"""

    def __init__(self, nf, heads, reduction):
        super().__init__()
        # 特征对齐：统一x1/x2特征分布（确保双时相特征可对比）
        self.align1 = nn.Conv2d(nf, nf, 1, bias=False)
        self.align2 = nn.Conv2d(nf, nf, 1, bias=False)

        # # 双时相注意力：基于f1、f2和差异生成权重（覆盖双向视角）
        # self.attn = nn.Sequential(
        #     nn.Conv2d(nf * 3, nf, 1),  # 输入：f1 + f2 + 差异特征（3*nf通道）
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(nf, nf, 1),
        #     nn.Sigmoid()  # 注意力权重（0~1，突出变化区域）
        # )
        # 替换为轻量化双向交叉注意力（核心优化）
        self.attn = LightWeight_DifferenceFeatureComplementaryAttention(nf, heads=heads, reduction=reduction)

        self.CBAM1 = CBAM.CBAMBlock(nf)
        self.CBAM2 = CBAM.CBAMBlock(nf)

        # 单独归一化：分别稳定两个增强特征的分布
        self.norm1 = LayerNorm2d(nf)  # 用于enhanced_f1
        self.norm2 = LayerNorm2d(nf)  # 用于enhanced_f2

        # 权重初始化
        nn.init.normal_(self.align1.weight, mean=0, std=0.01)
        nn.init.normal_(self.align2.weight, mean=0, std=0.01)
        initialize_weights(self.attn, 0.1)
        self.sigmoid = nn.Sigmoid()

        self.dino_diff_proj = nn.Conv2d(nf, nf, 1)

    # def spatial_difference(self, xA, xB):
    #     xA_flat = xA.permute(0, 2, 3, 1).reshape(-1, xA.size(1))
    #     xB_flat = xB.permute(0, 2, 3, 1).reshape(-1, xB.size(1))
    #     cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)
    #     cosine_sim = cosine_sim.view(xA.size(0), xA.size(2), xA.size(3))
    #     cosine_sim = cosine_sim.unsqueeze(1)
    #     c_weights = 1-self.sigmoid(cosine_sim)
    #     return c_weights
    def spatial_difference(self, xA, xB):
        B, C, H, W = xA.shape
        # 保留空间维度：(B, C, H, W) → (B, C, H*W)，每个位置对应一个特征向量
        xA_flat = xA.view(B, C, H*W)  # (B, C, HW)
        xB_flat = xB.view(B, C, H*W)  # (B, C, HW)
        # 按通道维度计算每个空间位置的余弦相似度（dim=1）
        cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)  # (B, HW)
        # 恢复空间维度：(B, 1, H, W)，与输入特征同尺寸
        cosine_sim = cosine_sim.view(B, 1, H, W)
        c_weights = 1 - self.sigmoid(cosine_sim)  # 差异越大，权重越高
        return c_weights

    def channel_difference(self, xA, xB):
        N, C, H, W = xA.shape
        xA_flat = xA.view(N, C, -1)
        xB_flat = xB.view(N, C, -1)
        cosine_sim = 1-self.sigmoid(F.cosine_similarity(xA_flat, xB_flat, dim=2))
        hw_weights = cosine_sim.unsqueeze(-1).unsqueeze(-1)
        return hw_weights

    def forward(self, feat_x1, feat_x2, dino_feat_x1, dino_feat_x2):
        """
        Args:
            feat_x1: x1单时相特征 [B, C, H, W]
            feat_x2: x2单时相特征 [B, C, H, W]
        Returns:
            enhanced_f1: 用f2增强后的x1特征 [B, C, H, W]
            enhanced_f2: 用f1增强后的x2特征 [B, C, H, W]
        """
        # 1. 特征对齐：消除双时相特征分布差异
        f1 = self.align1(feat_x1)  # x1特征对齐
        f2 = self.align2(feat_x2)  # x2特征对齐

        abs_diff = torch.abs(f1 - f2)

        dino_diff = torch.abs(dino_feat_x2 - dino_feat_x1)  # [B, nf, H, W]
        dino_diff_weight = F.sigmoid(self.dino_diff_proj(dino_diff))

        diff_feat = abs_diff * (1 + dino_diff_weight)

        # 计算空间权重和通道权重
        c_weights = self.spatial_difference(f1, f2)  # (2, 1, 32, 32)
        hw_weights = self.channel_difference(f1, f2)  # (2, 128, 1, 1)

        # 将 c_weights 扩展到与 hw_weights 相同的形状
        c_weights_expanded = c_weights.expand(-1, hw_weights.size(1), -1, -1)  # (2, 128, 32, 32)

        # 合并权重 (比如可以选择相乘，也可以进行加权平均)
        combined_weights = c_weights_expanded * hw_weights  # (2, 128, 32, 32)

        # # 对 xA 和 xB 进行加权处理
        # xA_weighted = f1 * combined_weights
        # xB_weighted = f2 * combined_weights
        #
        # # 2. 差异特征与双向注意力：捕捉双时相变化
          # 显式计算变化区域
        diff_feat_1 = diff_feat + diff_feat * combined_weights

        # attn_weight = self.attn(torch.cat([f1, f2, diff_feat], dim=1))  # 融合双时相视角生成权重
        #
        # # 3. 双向增强：用对方特征强化自身变化区域
        # # enhanced_f1：x1的基础上，用x2的变化区域特征增强
        # enhanced_f1 = f1 + attn_weight * f2
        # # enhanced_f2：x2的基础上，用x1的变化区域特征增强
        # enhanced_f2 = f2 + attn_weight * f1

        # 4. 轻量化双向交叉注意力（输出两个方向的权重）
        attn_weight_f1, attn_weight_f2 = self.attn(f1, f2, diff_feat_1)

        # 5. 双向增强
        enhanced_f1 = f1 + attn_weight_f1 * f2
        enhanced_f2 = f2 + attn_weight_f2 * f1

        # enhanced_f1 = self.CBAM1(enhanced_f1)
        # enhanced_f2 = self.CBAM2(enhanced_f2)

        # 4. 残差归一化：保留各自原始特征，稳定梯度
        # 对enhanced_f1：残差连接x1的原始特征（确保x1基础信息不丢失）
        enhanced_f1 = self.norm1(enhanced_f1 + feat_x1)
        # 对enhanced_f2：残差连接x2的原始特征（确保x2基础信息不丢失）
        enhanced_f2 = self.norm2(enhanced_f2 + feat_x2)

        enhanced_f1 = self.CBAM1(enhanced_f1)
        enhanced_f2 = self.CBAM2(enhanced_f2)

        return enhanced_f1, enhanced_f2, diff_feat_1 # 输出两个增强特征

class FusionAlign(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.align = nn.Sequential(
            DsBnRelu(dim*2, dim), CBAM.CBAMBlock(dim)
        )
    def forward(self, cnn_feat, dino_feat):
        fused_feat = self.align(torch.cat([cnn_feat, dino_feat], dim=1))  # 计算融合权重
        return fused_feat

class DsBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DsBnRelu, self).__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.kernel_size != 1:
            x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Pre_Post_BIE_Temporal_Fusion_Encoder(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2,
                 embed_dims=[64, 128, 256, 512], num_heads=[2, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 3, 6, 18],
                 sr_ratios=[8, 4, 2, 1], edge_channel=[32, 64, 128, 256], device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.edge_channel = edge_channel
        self.img_size = img_size

        self.stage_sizes = [64, 32, 16, 8]

        self.dino_wrapper = DINOV3Wrapper(
            weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            extract_ids=[5, 11, 17, 23],  # 提取4层特征，对应低→高语义
            device=device
        )

        self.dino_adapters = nn.ModuleList([
            DenseAdapterLite(
                in_dim=1024,
                out_dim=self.embed_dims[i],
                sizes=(self.stage_sizes[i],),  # 每个Stage单独适配尺寸
                bottleneck=32,  # 瓶颈层进一步轻量化（数据稀缺时减少参数）
                share=False
            ) for i in range(4)
        ])

        self.dino_fusion = nn.ModuleList([
            FusionAlign(dim=self.embed_dims[i]) for i in range(4)
        ])

        # ResNet34骨干
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        # 用于边缘提取的平均池化
        self.avg_pool_edge = nn.AvgPool2d((3, 3), stride=1, padding=1)

        # # Transformer Blocks（4个Stage）
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # cur = 0
        # self.block1 = nn.ModuleList([Block(
        #     dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
        #     attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
        #     norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        #     for i in range(depths[0])])
        # self.norm1 = norm_layer(embed_dims[0])
        # cur += depths[0]
        #
        # self.block2 = nn.ModuleList([Block(
        #     dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
        #     attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
        #     norm_layer=norm_layer, sr_ratio=sr_ratios[1])
        #     for i in range(depths[1])])
        # self.norm2 = norm_layer(embed_dims[1])
        # cur += depths[1]
        #
        # self.block3 = nn.ModuleList([Block(
        #     dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
        #     attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
        #     norm_layer=norm_layer, sr_ratio=sr_ratios[2])
        #     for i in range(depths[2])])
        # self.norm3 = norm_layer(embed_dims[2])
        # cur += depths[2]
        #
        # self.block4 = nn.ModuleList([Block(
        #     dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
        #     attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
        #     norm_layer=norm_layer, sr_ratio=sr_ratios[3])
        #     for i in range(depths[3])])
        # self.norm4 = norm_layer(embed_dims[3])
        #
        # # Patch Embedding
        # self.patch_embed1 = OverlapPatchEmbed(
        #     img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        # self.patch_embed2 = OverlapPatchEmbed(
        #     img_size=img_size//4, patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # self.patch_embed3 = OverlapPatchEmbed(
        #     img_size=img_size//8, patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        # self.patch_embed4 = OverlapPatchEmbed(
        #     img_size=img_size//16, patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # CNN Encoder（ResNet34的layer1~4）
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        self.mv2_config = get_config("x_small")
        # self.MV2= nn.ModuleList([
        #     ResLayers_Make(self.mv2_config, nf=embed_dims[i]) for i in range(4)
        # ])
        #
        # self.encoder1 = nn.ModuleList([
        #     self.MV2[i].layer_1 for i in range(4)
        # ])
        # self.encoder2 = nn.ModuleList([
        #     self.MV2[i].layer_2 for i in range(4)
        # ])
        # self.encoder3 = nn.ModuleList([
        #     self.MV2[i].layer_3 for i in range(4)
        # ])

        self.MV2 = ResLayers_Make(self.mv2_config)
        self.encoder1 = self.MV2.layer_1
        self.encoder2 = self.MV2.layer_2
        self.encoder3 = self.MV2.layer_3
        self.encoder4 = resnet.layer4

        # -------------------------- 新增：4个Stage的双向增强模块 --------------------------
        self.temporal_bie1 = Pre_Post_TemporalBIE(nf=embed_dims[0], heads=4, reduction=2)  # Stage1（64通道）
        self.temporal_bie2 = Pre_Post_TemporalBIE(nf=embed_dims[1], heads=4, reduction=2)  # Stage2（128通道）
        self.temporal_bie3 = Pre_Post_TemporalBIE(nf=embed_dims[2], heads=2, reduction=4)  # Stage3（256通道）
        self.temporal_bie4 = Pre_Post_TemporalBIE(nf=embed_dims[3], heads=2, reduction=8)  # Stage4（512通道）

        # 单时相内：Transformer+CNN特征融合（FA）
        # self.FA1 = nn.Conv2d(embed_dims[0]*2, embed_dims[0], kernel_size=1, stride=1, padding=0)
        # self.FA2 = nn.Conv2d(embed_dims[1]*2, embed_dims[1], kernel_size=1, stride=1, padding=0)
        # self.FA3 = nn.Conv2d(embed_dims[2]*2, embed_dims[2], kernel_size=1, stride=1, padding=0)
        # self.FA4 = nn.Conv2d(embed_dims[3]*2, embed_dims[3], kernel_size=1, stride=1, padding=0)
        # self.FA1 = FusionAlign(embed_dims[0])
        # self.FA2 = FusionAlign(embed_dims[1])
        # self.FA3 = FusionAlign(embed_dims[2])
        # self.FA4 = FusionAlign(embed_dims[3])

        # self.CBAM1 = CBAM.CBAMBlock(embed_dims[0])
        # self.CBAM2 = CBAM.CBAMBlock(embed_dims[1])
        # self.CBAM3 = CBAM.CBAMBlock(embed_dims[2])
        self.CBAM4 = CBAM.CBAMBlock(embed_dims[3])

        # 语义/边缘特征生成模块（复用原始逻辑）
        self.aspp = ASPP_v1(embed_dims[3])
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        # edge_channel=[32, 64, 128, 256]
        # embed_dims = [64, 128, 256, 512]
        self.convDown = nn.Conv2d(edge_channel[0], embed_dims[0], kernel_size=3, stride=2, padding=1)
        self.convDown1 = nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.convDown2 = nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1)
        self.convDown3 = nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(edge_channel[0], embed_dims[0], kernel_size=3, stride=1, padding=1)
        self.decoder4 = BilinearUp(embed_dims[3], embed_dims[2])
        self.decoder3 = BilinearUp(embed_dims[2], embed_dims[1])
        self.decoder2 = BilinearUp(embed_dims[1], embed_dims[0])
        self.decoder1 = BilinearUp(embed_dims[0], embed_dims[0])

        self.esa1 = Diff_SEM(sem_channel=embed_dims[3], edge_channel=edge_channel[2], out_channel=edge_channel[2], stride=4)
        self.esa2 = Diff_SEM(sem_channel=edge_channel[2], edge_channel=edge_channel[1], out_channel=edge_channel[1], stride=2)
        self.esa3 = Diff_SEM(sem_channel=edge_channel[1], edge_channel=edge_channel[1], out_channel=edge_channel[0], stride=2)

        # self.eem1 = SementicAddEdge(sem_channel=embed_dims[0], edge_channel=edge_channel[0], out_channel=edge_channel[0], stride=1)
        # self.eem2 = SementicAddEdge(sem_channel=embed_dims[0], edge_channel=edge_channel[1], out_channel=edge_channel[1], stride=1)
        # self.eem3 = SementicAddEdge(sem_channel=embed_dims[1], edge_channel=edge_channel[2], out_channel=edge_channel[2], stride=1)
        # self.eem4 = SementicAddEdge(sem_channel=embed_dims[2], edge_channel=edge_channel[3], out_channel=edge_channel[3], stride=1)
        # self.eem5 = SementicAddEdge(sem_channel=embed_dims[3], edge_channel=edge_channel[0], out_channel=edge_channel[0], stride=16)
        self.diff_proj_to_eem = nn.ModuleList([
            # 适配eem1：BIE Stage1（embed_dims[0]=64）→ eem1 out_channel（edge_channel[0]=32）
            nn.Conv2d(embed_dims[0], edge_channel[0], 1, 1),
            # 适配eem2：BIE Stage2（128）→ eem2 out_channel（64）
            nn.Conv2d(embed_dims[1], edge_channel[1], 1, 1),
            # 适配eem3：BIE Stage3（256）→ eem3 out_channel（128）
            nn.Conv2d(embed_dims[2], edge_channel[2], 1, 1),
            # 适配eem4：BIE Stage4（512）→ eem4 out_channel（256）
            nn.Conv2d(embed_dims[3], edge_channel[3], 1, 1),
            # 适配eem5：BIE Stage4（512）→ eem5 out_channel（32）
            nn.Conv2d(embed_dims[3], edge_channel[0], 1, 1)
        ])
        # 初始化投影层权重
        for proj in self.diff_proj_to_eem:
            nn.init.normal_(proj.weight, mean=0, std=0.01)

        self.eem1 = DiffSementicAddEdge(sem_channel=embed_dims[0], edge_channel=edge_channel[0],
                                    out_channel=edge_channel[0], stride=1, use_diff=True)
        self.eem2 = DiffSementicAddEdge(sem_channel=embed_dims[0], edge_channel=edge_channel[1],
                                    out_channel=edge_channel[1], stride=1, use_diff=True)
        self.eem3 = DiffSementicAddEdge(sem_channel=embed_dims[1], edge_channel=edge_channel[2],
                                    out_channel=edge_channel[2], stride=1, use_diff=True)
        self.eem4 = DiffSementicAddEdge(sem_channel=embed_dims[2], edge_channel=edge_channel[3],
                                    out_channel=edge_channel[3], stride=1, use_diff=True)
        self.eem5 = DiffSementicAddEdge(sem_channel=embed_dims[3], edge_channel=edge_channel[0],
                                    out_channel=edge_channel[0], stride=16, use_diff=True)

        # 权重初始化
        self.apply(self._init_weights)
        initialize_weights([self.dino_fusion, self.conv, self.convDown, self.convDown1, self.convDown2, self.convDown3], 0.1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _extract_single_phase_feat(self, x):
        """提取单个时相的原始特征（未增强，用于与另一时相交互）"""
        B = x.shape[0]
        x_cnn = x.clone()

        dino_feats_raw = self.dino_wrapper(x)

        # 浅层边缘特征
        x_cnn = self.firstconv(x_cnn)
        x_cnn = self.firstbn(x_cnn)
        # x_cnn_edge = self.firstrelu(x_cnn)  # 单时相边缘特征
        # x_cnn = self.firstmaxpool(x_cnn_edge)
        x_cnn_avgpool_use = x_cnn.clone()
        x_cnn_avg= self.avg_pool_edge(x_cnn_avgpool_use)
        x_cnn_edge_raw = x_cnn - x_cnn_avg  # edge = x - avg_pool(x)
        x_cnn_edge = self.firstrelu(x_cnn_edge_raw)

        x_cnn = self.firstrelu(x_cnn)
        x_cnn = self.firstmaxpool(x_cnn)

        _, _, H1, W1 = x_cnn.shape
        # Stage 1
        # x_tr1, H1, W1 = self.patch_embed1(x_tr)
        # for blk in self.block1:
        #     x_tr1 = blk(x_tr1, H1, W1)
        # x_tr1 = self.norm1(x_tr1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn1 = self.encoder1(x_cnn)
        # fa1 = self.FA1(torch.cat([x_tr1, x_cnn1], dim=1))
        # fa1 = self.FA1(x_tr1, x_cnn1)
        # fa1 = self.CBAM1(fa1)

        dino_feat1 = self.dino_adapters[0]([dino_feats_raw[0]])[0]
        fa1 = self.dino_fusion[0](x_cnn1, dino_feat1)

        _, _, H2, W2 = fa1.shape
        # Stage 2
        # x_tr2, H2, W2 = self.patch_embed2(fa1)
        # for blk in self.block2:
        #     x_tr2 = blk(x_tr2, H2, W2)
        # x_tr2 = self.norm2(x_tr2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn2 = self.encoder2(fa1)
        # fa2 = self.FA2(torch.cat([x_tr2, x_cnn2], dim=1))
        # fa2 = self.FA2(x_tr2, x_cnn2)
        # fa2 = self.CBAM2(fa2)

        dino_feat2 = self.dino_adapters[1]([dino_feats_raw[1]])[0]  # [B, 128, 32, 32]
        fa2 = self.dino_fusion[1](x_cnn2, dino_feat2)

        _, _, H3, W3 = fa2.shape
        # Stage 3
        # x_tr3, H3, W3 = self.patch_embed3(fa2)
        # for blk in self.block3:
        #     x_tr3 = blk(x_tr3, H3, W3)
        # x_tr3 = self.norm3(x_tr3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn3 = self.encoder3(fa2)

        # fa3 = self.FA3(x_tr3, x_cnn3)
        # fa3 = self.CBAM3(fa3)

        dino_feat3 = self.dino_adapters[2]([dino_feats_raw[2]])[0]  # [B, 256, 16, 16]
        fa3 = self.dino_fusion[2](x_cnn3, dino_feat3)

        _, _, H4, W4 = fa3.shape
        # Stage 4
        # x_tr4, H4, W4 = self.patch_embed4(fa3)
        # for blk in self.block4:
        #     x_tr4 = blk(x_tr4, H4, W4)
        # x_tr4 = self.norm4(x_tr4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn4 = self.encoder4(fa3)
        # fa4 = self.FA4(x_tr4, x_cnn4)
        # fa4 = self.CBAM4(fa4)

        dino_feat4 = self.dino_adapters[3]([dino_feats_raw[3]])[0]  # [B, 512, 8, 8]
        fa4 = self.dino_fusion[3](x_cnn4, dino_feat4)

        return [fa1, fa2, fa3, fa4], x_cnn_edge, [H1, W1, H2, W2, H3, W3, H4, W4], [dino_feat1, dino_feat2, dino_feat3, dino_feat4]

    def _generate_sem_edge(self, fa_list, edge_feat, diff_feat, sizes):
        """基于增强后的FA特征生成语义和边缘特征（复用原始逻辑）"""
        H1, W1, H2, W2, H3, W3, H4, W4 = sizes
        fa1, fa2, fa3, fa4 = fa_list
        edge_channel = self.edge_channel  # [32,64,128,256]

        # ASPP语义增强
        e4 = self.aspp(fa4)
        # e4 = self.CBAM4(e4)

        # Edge Self-Attention边缘自注意力模块
        # 用于获取边缘特征
        # ESA：边缘-语义交互
        fa2_edge = fa2 - self.avg_pool_edge(fa2)  # edge = x - avg_pool(x)
        fa2_edge = self.firstrelu(fa2_edge)

        fa1_edge = fa1 - self.avg_pool_edge(fa1)  # edge = x - avg_pool(x)
        fa1_edge = self.firstrelu(fa1_edge)

        x_cat2_addSem = self.esa1(e4, fa2_edge, diff_feat[1])
        x_cat1_addSem = self.esa2(x_cat2_addSem, fa1_edge, diff_feat[0])
        x_cnn_edge_addSem = self.esa3(x_cat1_addSem, edge_feat)
        edge_res = [x_cat2_addSem, x_cat1_addSem, x_cnn_edge_addSem]

        # IMD：特征下采样与残差连接
        x_cnn_edge_addSem_down = self.leakyRelu(self.convDown(x_cnn_edge_addSem))
        x_cat1_down = self.leakyRelu(self.convDown1(fa1 + x_cnn_edge_addSem_down))
        x_cat2_down = self.leakyRelu(self.convDown2(fa2 + x_cat1_down))
        x_cat3_down = self.leakyRelu(self.convDown3(fa3 + x_cat2_down))

        d4 = self.decoder4(e4 + x_cat3_down) + fa3 + x_cat2_down
        d3 = self.decoder3(d4) + fa2 + x_cat1_down
        d2 = self.decoder2(d3) + fa1 + x_cnn_edge_addSem_down
        d1 = self.decoder1(d2) + self.conv(x_cnn_edge_addSem)

        # EEM：语义-边缘融合
        x_cnn_edge_addSem_down2 = self.leakyRelu(self.convDown1(x_cnn_edge_addSem_down))
        x_cnn_edge_addSem_down3 = self.leakyRelu(self.convDown2(x_cnn_edge_addSem_down2))

        def _prepare_diff_for_eem(diff_feat_bie, proj_layer, target_size):
            diff_feat_proj = proj_layer(diff_feat_bie)  # 用预定义投影层
            diff_feat_aligned = F.interpolate(
                diff_feat_proj, size=target_size, mode='bilinear', align_corners=False
            )
            return diff_feat_aligned

        diff_eem1 = _prepare_diff_for_eem(diff_feat[0], self.diff_proj_to_eem[0], (H1, W1))
        diff_eem2 = _prepare_diff_for_eem(diff_feat[1], self.diff_proj_to_eem[1], (H2, W2))
        diff_eem3 = _prepare_diff_for_eem(diff_feat[2], self.diff_proj_to_eem[2], (H3, W3))
        diff_eem4 = _prepare_diff_for_eem(diff_feat[3], self.diff_proj_to_eem[3], (H4, W4))
        diff_eem5 = _prepare_diff_for_eem(diff_feat[3], self.diff_proj_to_eem[4], (H1, W1))

        sem1 = self.eem1(d1, x_cnn_edge_addSem, diff_eem1)
        sem2 = self.eem2(d2, x_cnn_edge_addSem_down, diff_eem2)
        sem3 = self.eem3(d3, x_cnn_edge_addSem_down2, diff_eem3)
        sem4 = self.eem4(d4, x_cnn_edge_addSem_down3, diff_eem4)
        sem5 = self.eem5(e4, x_cnn_edge_addSem, diff_eem5)

        # sem1 = self.eem1(d1, x_cnn_edge_addSem)
        # sem2 = self.eem2(d2, x_cnn_edge_addSem_down)
        # sem3 = self.eem3(d3, x_cnn_edge_addSem_down2)
        # sem4 = self.eem4(d4, x_cnn_edge_addSem_down3)
        # sem5 = self.eem5(e4, x_cnn_edge_addSem)
        sem_res = [sem1, sem2, sem3, sem4, sem5]

        return sem_res, edge_res

    def forward(self, x1, x2):
        """
        输出：x1和x2各自的语义特征和边缘特征（原始格式）
        返回：(x1_sem_res, x1_edge_res), (x2_sem_res, x2_edge_res)
        """
        # 1. 分别提取x1和x2的原始特征（未增强）
        x1_fa_raw, x1_edge_raw, sizes, x1_dino_feats = self._extract_single_phase_feat(x1)
        x2_fa_raw, x2_edge_raw, _, x2_dino_feats = self._extract_single_phase_feat(x2)
        H1, W1, H2, W2, H3, W3, H4, W4 = sizes

        # 2. 逐Stage双向增强：x1的FA特征被x2增强，x2的FA特征被x1增强
        enhanced_x1_fa1, enhanced_x2_fa1, diff_feat_1 = self.temporal_bie1(x1_fa_raw[0], x2_fa_raw[0], x1_dino_feats[0], x2_dino_feats[0])  # Stage1
        enhanced_x1_fa2, enhanced_x2_fa2, diff_feat_2 = self.temporal_bie2(x1_fa_raw[1], x2_fa_raw[1], x1_dino_feats[1], x2_dino_feats[1])  # Stage2
        enhanced_x1_fa3, enhanced_x2_fa3, diff_feat_3 = self.temporal_bie3(x1_fa_raw[2], x2_fa_raw[2], x1_dino_feats[2], x2_dino_feats[2])  # Stage3
        enhanced_x1_fa4, enhanced_x2_fa4, diff_feat_4 = self.temporal_bie4(x1_fa_raw[3], x2_fa_raw[3], x1_dino_feats[3], x2_dino_feats[3])  # Stage4

        # 增强后的FA特征列表
        x1_fa_enhanced = [enhanced_x1_fa1, enhanced_x1_fa2, enhanced_x1_fa3, enhanced_x1_fa4]
        x2_fa_enhanced = [enhanced_x2_fa1, enhanced_x2_fa2, enhanced_x2_fa3, enhanced_x2_fa4]
        diff_feat_stage = [diff_feat_1, diff_feat_2, diff_feat_3, diff_feat_4]

        # 3. 基于增强后的特征生成x1和x2各自的语义/边缘特征（保留原始输出格式）
        x1_sem_res, x1_edge_res = self._generate_sem_edge(x1_fa_enhanced, x1_edge_raw, diff_feat_stage, sizes)
        x2_sem_res, x2_edge_res = self._generate_sem_edge(x2_fa_enhanced, x2_edge_raw, diff_feat_stage, sizes)

        # 4. 按原始格式返回：x1的[sem_res, edge_res]和x2的[sem_res, edge_res]
        return [x1_sem_res, x1_edge_res], [x2_sem_res, x2_edge_res], diff_feat_stage

# Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def chunk_attention(q, k, v, chunk_size):
    B, H, N, D = q.shape
    out = []

    chunk_softmax = nn.Softmax(dim=-1)
    for i in range(0, N, chunk_size):
        q_chunk = q[:, :, i:i + chunk_size, :]  # (B, H, chunk, D)
        k_chunk = k[:, :, :, i:i + chunk_size]  # (B, H, D, chunk)
        v_chunk = v[:, :, :, i:i + chunk_size]  # (B, H, D, chunk)

        # 缩放点积注意力（减小数值范围，避免梯度爆炸）
        scale = torch.sqrt(torch.tensor(D, dtype=torch.float32, device=q.device))
        energy = torch.matmul(q_chunk, k_chunk) / scale  # (B, H, chunk, chunk)
        attn = chunk_softmax(energy)

        out_chunk = torch.matmul(v_chunk, attn.permute(0, 1, 3, 2))  # (B, H, D, chunk)
        out.append(out_chunk)
    return torch.cat(out, dim=-1)

class Atten_Cross(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Cross, self).__init__()
        self.chanel_in = in_dim

        # self.num_heads = num_heads  # 多头拆分维度
        # self.chunk_size = chunk_size  # 分块大小（控制单块显存）
        # self.head_dim = in_dim // num_heads
        # assert self.head_dim * num_heads == in_dim, "in_dim必须能被num_heads整除"

        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, cond):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()
        # N = height * width

        # guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True) # map 2,1,128,128 -> 2,1,16,16
        #
        # guiding_map = F.sigmoid(guiding_map0)

        # query = self.query_conv(x).view(m_batchsize, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        # key = self.key_conv(cond).view(m_batchsize, self.num_heads, self.head_dim, N).permute(0, 1, 2, 3)  # (B, heads, head_dim, N)
        # value = self.value_conv(cond).view(m_batchsize, self.num_heads, self.head_dim, N)  # (B, heads, head_dim, N)

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

        # out = chunk_attention(query, key, value, self.chunk_size)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

class Atten_Spa(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Spa, self).__init__()
        self.chanel_in = in_dim

        # self.num_heads = num_heads
        # self.chunk_size = chunk_size
        # self.head_dim = in_dim // num_heads
        # assert self.head_dim * num_heads == in_dim, "in_dim必须能被num_heads整除"

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()
        # N = height * width

        # query = self.query_conv(x).view(m_batchsize, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        # key = self.key_conv(x).view(m_batchsize, self.num_heads, self.head_dim, N).permute(0, 1, 2, 3)
        # value = self.value_conv(x).view(m_batchsize, self.num_heads, self.head_dim, N)

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

        # out = chunk_attention(query, key, value, self.chunk_size)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

class Diff_HSCA(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.concat_dim = dim * 2

        self.proj = nn.Conv2d(self.concat_dim, dim, kernel_size=1, padding=0)

        self.self_attn = Atten_Spa(in_dim=dim)

        # self.cross_attn = Atten_Cross(in_dim=dim)
        self.cross_attn = Atten_Cross(in_dim=dim)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 先把diff_prior投影到dim通道（兼容任意输入通道）
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),  # 最终生成单通道掩码
            nn.Sigmoid()  # 0~1掩码，控制差异区域贡献
        )

        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat1, feat2, diff_prior=None):
        """
        完全还原HSANet流程：
        feat1: x1（T1）的Sem Stage特征 [B, 32, H, W]
        feat2: x2（T2）的Sem Stage特征 [B, 32, H, W]
        返回：增强后的双时相融合特征 [B, 32, H, W]
        """
        concat_feat = torch.cat([feat1, feat2], dim=1)  # [B, 64, H, W]


        proj_feat = self.proj(concat_feat)  # [B, 32, H, W]

        self_attn_feat = self.self_attn(proj_feat)  # [B, 32, H, W]

        if diff_prior is not None:

            diff_prior_aligned = F.interpolate(
                diff_prior, size=self_attn_feat.shape[2:], mode='bilinear', align_corners=False
            )

            diff_gate = self.gate(diff_prior_aligned)

            guided_feat = self_attn_feat * diff_gate
        else:
            guided_feat = self_attn_feat

        # cross_attn_feat = self.cross_attn(proj_feat, self_attn_feat)  # [B, 32, H, W]
        cross_attn_feat = self.cross_attn(proj_feat, guided_feat)

        final_feat = self.norm(cross_attn_feat + proj_feat)
        final_feat = self.relu(final_feat)

        return final_feat

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

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class CAMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape

        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B2, N2, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CABlock(nn.Module):
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.attn = CrossAttention(dim1, dim2, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = CAMlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

class SequenceReducer(nn.Module):
    """序列约简模块：通过卷积聚合减少序列长度，同时压缩通道（贴合ChangeViT轻量设计）"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        # 核心：stride卷积聚合空间维度（序列长度 = 原长度 / stride²）
        self.spatial_reduce = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False  # 深度卷积，轻量高效
        )
        # 通道压缩：对齐ViT特征通道（32）
        self.channel_compress = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: 低尺度细节特征 → [B, C_in, H, W]（高分辨率，如128×128）
        返回：约简后的特征 → [B, C_out, H/stride, W/stride]（低分辨率，如64×64）
        """
        # 空间聚合：序列长度减少4倍（stride=2）
        x_reduced = self.spatial_reduce(x)
        # 通道压缩：对齐ViT特征通道
        x_compressed = self.channel_compress(x_reduced)
        return self.relu(self.bn(x_compressed))

class FeatureInjector(nn.Module):
    def __init__(self, dim1=32, dim2=[64, 128, 256], num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.sequence_reducers = nn.ModuleList([
            SequenceReducer(dim2[0], dim1, stride=2),
            SequenceReducer(dim2[1], dim1, stride=2),
            SequenceReducer(dim2[2], dim1, stride=2),
        ])

        self.c2_c5 = CABlock(dim1, dim1, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c3_c5 = CABlock(dim1, dim1, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c4_c5 = CABlock(dim1, dim1, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        # self.cross_attns = nn.ModuleList([
        #     CABlock(dim1, dim1, num_heads, mlp_ratio, qkv_bias, drop) for _ in range(3)
        # ])
        self.fuse = nn.Conv2d(dim1*3, dim1, 1, bias=False)  # 融合3个跨注意力输出
        self.bn = nn.BatchNorm2d(dim1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, c2, c3, c4, c5):
        B, C, H, W = c5.shape
        # 展平为序列特征（适配跨注意力）

        c2_reduced = self.sequence_reducers[0](c2)
        c3_reduced = self.sequence_reducers[1](c3)
        c4_reduced = self.sequence_reducers[2](c4)

        # c2_aligned = F.interpolate(c2_reduced, (H5, W5), mode='bilinear', align_corners=False)
        # c3_aligned = F.interpolate(c3_reduced, (H5, W5), mode='bilinear', align_corners=False)
        # c4_aligned = F.interpolate(c4_reduced, (H5, W5), mode='bilinear', align_corners=False)

        c5_flat = rearrange(c5, 'b c h w -> b (h w) c')
        c2_flat = rearrange(c2_reduced, 'b c h w -> b (h w) c')
        c3_flat = rearrange(c3_reduced, 'b c h w -> b (h w) c')
        c4_flat = rearrange(c4_reduced, 'b c h w -> b (h w) c')

        # 跨注意力注入：c5为query，c2/c3/c4为key/value
        _c2 = self.c2_c5(c5_flat, c2_flat)
        _c2 = rearrange(_c2, 'b (h w) c -> b c h w', h=H, w=W)

        _c3 = self.c3_c5(c5_flat, c3_flat)
        _c3 = rearrange(_c3, 'b (h w) c -> b c h w', h=H, w=W)

        _c4 = self.c4_c5(c5_flat, c4_flat)
        _c4 = rearrange(_c4, 'b (h w) c -> b c h w', h=H, w=W)

        # 融合注入后的特征
        fused_c5 = self.fuse(torch.cat([_c2, _c3, _c4], dim=1))
        fused = self.bn(fused_c5)
        return F.relu(fused + c5)

class LearnableSoftMorph(nn.Module):
    def __init__(self, k_open=3, k_close=5, tau=0.05):
        super().__init__()
        assert k_open >= 1 and k_close >= 1
        self.k_open, self.k_close = k_open, k_close
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))

        def make_kernel(k):
            n = k * k
            return nn.Parameter(torch.zeros(1, n))

        self.we_open_erode   = make_kernel(k_open)
        self.wd_open_dilate  = make_kernel(k_open)
        self.wd_close_dilate = make_kernel(k_close)
        self.we_close_erode  = make_kernel(k_close)

        # after sigomid: α ∈ [0,1]
        self.alpha_raw = nn.Parameter(torch.tensor(-5.0))

    @staticmethod
    def _logsumexp_pool(x_cols, w, tau):
        # x_cols: [B, 1, K, HW]; w: [1, K]; tau: scalar
        # soft max: tau * logsumexp( (x + w)/tau, dim=2 )
        z = torch.logsumexp((x_cols + w.unsqueeze(-1)) / tau, dim=2) * tau
        return z  # [B, 1, HW]

    def _soft_dilate(self, x, k, w, tau):
        if k <= 1: return x
        B, _, H, W = x.shape
        pad = k // 2
        cols = F.unfold(x, k, padding=pad)           # [B, 1*K*K, HW]
        cols = cols.view(B, 1, k*k, H*W)             # [B, 1, K*K, HW]
        z = self._logsumexp_pool(cols, w, tau)       # [B, 1, HW]
        return z.view(B, 1, H, W)

    def _soft_erode(self, x, k, w, tau):
        # min(x+s) = -softmax( -x + (-s) )
        return -self._soft_dilate(-x, k, -w, tau)

    def forward(self, logit_2ch):
        _, C, _, _ = logit_2ch.shape
        assert C == 2, "Expect 2-channel logits for binary segmentation."

        p_fg = F.softmax(logit_2ch, dim=1)[:, 1:2]  # [B,1,H,W]
        tau = torch.exp(self.log_tau).clamp_min(1e-4)

        # Open: erode -> dilate
        p = self._soft_erode (p_fg, self.k_open,  self.we_open_erode,   tau)
        p = self._soft_dilate(p,    self.k_open,  self.wd_open_dilate,  tau)
        # Close: dilate -> erode
        p = self._soft_dilate(p,    self.k_close, self.wd_close_dilate, tau)
        p = self._soft_erode (p,    self.k_close, self.we_close_erode,  tau)

        fg_logit_refined = torch.logit(p.clamp(1e-6, 1-1e-6), eps=1e-6)

        alpha = torch.sigmoid(self.alpha_raw)  # [0,1]
        out = logit_2ch.clone()
        out[:, 1:2] = out[:, 1:2] + alpha * (fg_logit_refined - out[:, 1:2])

        return out

class Decoder(nn.Module):
    def __init__(self, num_classes=1, embedding_dim=32, output_nc=2, edge_channel=[32, 64, 128, 256],
                 scale_dims = [64, 128, 256]):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.num_classes = num_classes
        self.edge_channel = edge_channel
        self.scale_dims = scale_dims


        # # convolutional Difference Modules
        self.diff_c5 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        # self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        self.ef3 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)
        self.ef2 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)
        self.ef1 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)

        self.downconv1 = nn.Conv2d(in_channels=self.edge_channel[1], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        self.downconv2 = nn.Conv2d(in_channels=self.edge_channel[2], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        self.downconv3 = nn.Conv2d(in_channels=self.edge_channel[3], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        self.downconv4 = nn.Conv2d(in_channels=self.edge_channel[0], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        # # Final linear fusion layer
        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embedding_dim * 5, out_channels=self.embedding_dim,
        #               kernel_size=1),
        #     nn.BatchNorm2d(self.embedding_dim)
        # )
        self.edge_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.edgeOut = BilinearUp(self.embedding_dim, self.embedding_dim, 2)
        self.finalcat_edge = nn.Conv2d(self.embedding_dim, num_classes, 1, 1)

        self.diff_hsca = nn.ModuleList([
            Diff_HSCA(dim=self.scale_dims[i]) for i in range(3)  # 每个尺度用自己的通道数
        ])

        self.hsan_fuse = BasicConv2d(2 * self.embedding_dim, self.embedding_dim, 3, 1, 1)
        # self.hsan_fuse = nn.ModuleList([
        #     BasicConv2d(self.embedding_dim + self.scale_dims[2], self.embedding_dim, 3, 1, 1),
        #     BasicConv2d(self.embedding_dim + self.scale_dims[1], self.embedding_dim, 3, 1, 1),
        #     BasicConv2d(self.embedding_dim + self.scale_dims[0], self.embedding_dim, 3, 1, 1),
        #     BasicConv2d(self.embedding_dim + self.embedding_dim, self.embedding_dim, 3, 1, 1)
        # ])

        # sem4(256) → sem3(128) 通道压缩
        self.channel_compress_4to3 = nn.Conv2d(self.scale_dims[2], self.scale_dims[1], 1, bias=False)
        # sem3(128) → sem2(64) 通道压缩
        self.channel_compress_3to2 = nn.Conv2d(self.scale_dims[1], self.scale_dims[0], 1, bias=False)
        # sem2(64) → sem1(32) 通道压缩（sem1是embedding_dim=32）
        self.channel_compress_2to1 = nn.Conv2d(self.scale_dims[0], self.embedding_dim, 1, bias=False)

        self.feature_injector = FeatureInjector(
            dim1=self.embedding_dim,  # sem5_enh通道数
            dim2=self.scale_dims,  # sem2(64)/sem3(128)/sem4(256)通道数
            num_heads=4,  # 跨注意力头数（适配dim1=32，32//4=8头_dim）
            mlp_ratio=4.,
            qkv_bias=True
        )

        self.scale_compress4 = nn.Conv2d(self.embedding_dim + self.scale_dims[2], self.embedding_dim, 1,
                                         bias=False)  # fuse4+sem4_enh: 32+256=288→32
        self.scale_compress3 = nn.Conv2d(self.embedding_dim + self.scale_dims[1], self.embedding_dim, 1,
                                         bias=False)  # fuse3+sem3_enh: 32+128=160→32
        self.scale_compress2 = nn.Conv2d(self.embedding_dim + self.scale_dims[0], self.embedding_dim, 1,
                                         bias=False)  # fuse2+sem2_enh: 32+64=96→32

        for m in [self.scale_compress4, self.scale_compress3, self.scale_compress2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')  # 适配LeakyReLU
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)

        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = nn.Sequential(
            ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1),
        )
        self.refiner = LearnableSoftMorph(3, 5)

        self.change_edge = ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs1, inputs2, bie_diff_prior):
        c1_1, c2_1, c3_1, c4_1, c5_1 = inputs1[0]
        c1_2, c2_2, c3_2, c4_2, c5_2 = inputs2[0]

        e1_1, e2_1, e3_1 = inputs1[1]
        e1_2, e2_2, e3_2 = inputs2[1]

        outputs = []

        # Edge Decoder

        # Edge Stage 1
        e1_1_down = self.downconv2(e1_1)
        e1_2_down = self.downconv2(e1_2)
        _e1 = self.ef1(e1_1_down, e1_2_down)
        _e1_up = resize(_e1, size=e3_2.size()[2:], mode='bilinear', align_corners=False)

        # Edge Stage 2
        e2_1_down = self.downconv1(e2_1)
        e2_2_down = self.downconv1(e2_2)
        _e2 = self.ef2(e2_1_down, e2_2_down) + F.interpolate(_e1, scale_factor=2, mode="bilinear")
        _e2_up = resize(_e2, size=e3_2.size()[2:], mode='bilinear', align_corners=False)

        # Edge Stage 3
        e3_1_down = self.downconv4(e3_1)
        e3_2_down = self.downconv4(e3_2)
        _e3 = self.ef3(e3_1_down, e3_2_down) + F.interpolate(_e2, scale_factor=2, mode="bilinear")
        # EMFF
        _e = self.leakyRelu(self.edge_fuse(torch.cat((_e1_up, _e2_up, _e3), dim=1)))

        e_out = self.edgeOut(_e)
        e_out = self.finalcat_edge(e_out)
        outputs.append(e_out)

        c5_1_down = self.downconv4(c5_1)
        c5_2_down = self.downconv4(c5_2)
        sem5_enh = self.diff_c5(torch.cat((c5_1_down, c5_2_down), dim=1))

        # sem4
        sem4_enh = self.diff_hsca[2](c4_1, c4_2, bie_diff_prior[2])

        # sem3
        sem3_enh = self.diff_hsca[1](c3_1, c3_2, bie_diff_prior[1])
        sem4_compress = self.channel_compress_4to3(sem4_enh)  # 256→128
        sem4_interp = F.interpolate(sem4_compress, size=sem3_enh.shape[2:], mode='bilinear', align_corners=False)
        sem3_enh = sem3_enh + sem4_interp

        # sem2
        sem2_enh = self.diff_hsca[0](c2_1, c2_2, bie_diff_prior[0])
        sem3_compress = self.channel_compress_3to2(sem3_enh)  # 128→64
        sem3_interp = F.interpolate(sem3_compress, size=sem2_enh.shape[2:], mode='bilinear', align_corners=False)
        sem2_enh = sem2_enh + sem3_interp

        # sem1
        c1_1_down = self.downconv4(c1_1)
        c1_2_down = self.downconv4(c1_2)
        sem1_enh = self.diff_c1(torch.cat((c1_1_down, c1_2_down), dim=1))
        sem2_compress = self.channel_compress_2to1(sem2_enh)  # 64→32
        sem2_interp = F.interpolate(sem2_compress, size=sem1_enh.shape[2:], mode='bilinear', align_corners=False)
        sem1_enh = sem1_enh + sem2_interp


        # # sem5→sem4融合
        # fuse4 = self.hsan_fuse[0](torch.cat([F.interpolate(sem5_enh, size=sem4_enh.shape[2:], mode='bilinear', align_corners=False),sem4_enh], dim=1))
        # # sem4→sem3融合
        # fuse3 = self.hsan_fuse[1](torch.cat([F.interpolate(fuse4, size=sem3_enh.shape[2:], mode='bilinear', align_corners=False),sem3_enh], dim=1))
        # # sem3→sem2融合
        # fuse2 = self.hsan_fuse[2](torch.cat([F.interpolate(fuse3, size=sem2_enh.shape[2:], mode='bilinear', align_corners=False),sem2_enh], dim=1))
        # # sem2→sem1融合
        # fuse1 = self.hsan_fuse[3](torch.cat([F.interpolate(fuse2, size=sem1_enh.shape[2:], mode='bilinear', align_corners=False),sem1_enh], dim=1))
        fused_high = self.feature_injector(sem2_enh, sem3_enh, sem4_enh, sem5_enh)



        fused = self.hsan_fuse(torch.cat([sem1_enh, fused_high], dim=1))

        c = self.convd1x(fused)
        c = self.dense_1x(c)
        cp = self.change_probability(c)
        # cp = self.refiner(cp)

        outputs.append(cp)
        return outputs

class Decoder11(nn.Module):
    def __init__(self, num_classes=1, embedding_dim=32, output_nc=2, edge_channel=[32, 64, 128, 256], scale_dims = [64, 128, 256]):
        super(Decoder11, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.num_classes = num_classes
        self.edge_channel = edge_channel

        # convolutional Difference Modules
        self.diff_c5 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        self.ef3 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)
        self.ef2 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)
        self.ef1 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)

        self.downconv1 = nn.Conv2d(in_channels=self.edge_channel[1], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        self.downconv2 = nn.Conv2d(in_channels=self.edge_channel[2], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        self.downconv3 = nn.Conv2d(in_channels=self.edge_channel[3], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        self.downconv4 = nn.Conv2d(in_channels=self.edge_channel[0], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 5, out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        self.edge_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.edgeOut = BilinearUp(self.embedding_dim, self.embedding_dim, 2)
        self.finalcat_edge = nn.Conv2d(self.embedding_dim, num_classes, 1, 1)

        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)

        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.change_edge = ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1)

        self.refiner = LearnableSoftMorph(3, 5)

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1, c5_1 = inputs1[0]
        c1_2, c2_2, c3_2, c4_2, c5_2 = inputs2[0]

        e1_1, e2_1, e3_1 = inputs1[1]
        e1_2, e2_2, e3_2 = inputs2[1]

        outputs = []

        # Edge Stage 1
        e1_1_down = self.downconv2(e1_1)
        e1_2_down = self.downconv2(e1_2)
        _e1 = self.ef1(e1_1_down, e1_2_down)
        _e1_up = resize(_e1, size=e3_2.size()[2:], mode='bilinear', align_corners=False)

        # Edge Stage 2
        e2_1_down = self.downconv1(e2_1)
        e2_2_down = self.downconv1(e2_2)
        _e2 = self.ef2(e2_1_down, e2_2_down) + F.interpolate(_e1, scale_factor=2, mode="bilinear")
        _e2_up = resize(_e2, size=e3_2.size()[2:], mode='bilinear', align_corners=False)

        # Edge Stage 3
        e3_1_down = self.downconv4(e3_1)
        e3_2_down = self.downconv4(e3_2)
        _e3 = self.ef3(e3_1_down, e3_2_down) + F.interpolate(_e2, scale_factor=2, mode="bilinear")
        # EMFF
        _e = self.leakyRelu(self.edge_fuse(torch.cat((_e1_up, _e2_up, _e3), dim=1)))

        e_out = self.edgeOut(_e)
        e_out = self.finalcat_edge(e_out)
        outputs.append(e_out)

        # Sem Stage 5
        c5_1_down = self.downconv4(c5_1)
        c5_2_down = self.downconv4(c5_2)
        _c5 = self.diff_c5(torch.cat((c5_1_down, c5_2_down), dim=1))
        _c5_up = resize(_c5, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Sem Stage 4
        c4_1_down = self.downconv3(c4_1)
        c4_2_down = self.downconv3(c4_2)
        _c4 = self.diff_c4(torch.cat((c4_1_down, c4_2_down), dim=1))
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Sem Stage 3
        c3_1_down = self.downconv2(c3_1)
        c3_2_down = self.downconv2(c3_2)
        _c3 = self.diff_c3(torch.cat((c3_1_down, c3_2_down), dim=1)) + F.interpolate(_c4, scale_factor=2,
                                                                                     mode="bilinear")
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Sem Stage 2
        c2_1_down = self.downconv1(c2_1)
        c2_2_down = self.downconv1(c2_2)
        _c2 = self.diff_c2(torch.cat((c2_1_down, c2_2_down), dim=1)) + F.interpolate(_c3, scale_factor=2,
                                                                                     mode="bilinear")
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Sem Stage 1
        c1_1_down = self.downconv4(c1_1)
        c1_2_down = self.downconv4(c1_2)
        _c1 = self.diff_c1(torch.cat((c1_1_down, c1_2_down), dim=1)) + F.interpolate(_c2, scale_factor=2,
                                                                                     mode="bilinear")

        # SEMM
        _c = self.linear_fuse(torch.cat((_c5_up, _c4_up, _c3_up, _c2_up, _c1), dim=1))

        c = self.convd1x(_c)
        c = self.dense_1x(c)
        cp = self.change_probability(c)
        # cp = self.refiner(cp)

        outputs.append(cp)
        return outputs


class BIE_EdgeNet_DINO(nn.Module):
    def __init__(self, img_size=256, input_nc=3, output_nc=2, embed_dim=32, num_classes=2):
        super(BIE_EdgeNet_DINO, self).__init__()
        # Encoder
        # 编码器嵌入维度
        self.embed_dims = [64, 128, 256, 512]
        self.depths = [3, 3, 4, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1
        self.num_classes = num_classes
        self.img_size = img_size

        self.FE_IMD = Pre_Post_BIE_Temporal_Fusion_Encoder(img_size=self.img_size,
                              patch_size=3,
                              in_chans=input_nc,
                              num_classes=self.num_classes,
                              embed_dims=[64, 128, 256, 512],
                              num_heads=[2, 2, 4, 8],
                              mlp_ratios=[4, 4, 4, 4],
                              qkv_bias=True,
                              qk_scale=None,
                              drop_rate=0.,
                              attn_drop_rate=0.,
                              drop_path_rate=0.,
                              norm_layer=nn.LayerNorm,
                              depths=[3, 3, 6, 18],
                              sr_ratios=[8, 4, 2, 1],
                              edge_channel=[32, 64, 128, 256])

        self.CD_ED = Decoder11(num_classes=1,
                             embedding_dim=self.embedding_dim,
                             output_nc=output_nc,
                             edge_channel=[32, 64, 128, 256],
                             scale_dims = [64, 128, 256])

    def forward(self, x1, x2):
        fx1, fx2, bie_diff_prior = self.FE_IMD(x1, x2)

        # cp = self.CD_ED(fx1, fx2, bie_diff_prior)
        cp = self.CD_ED(fx1, fx2)

        return cp


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = BIE_EdgeNet_DINO(img_size=256, input_nc=3, output_nc=2, embed_dim=32, num_classes=2).to(device)
    x1 = torch.randn(1, 3, 256, 256).to(device)
    x2 = torch.randn(1, 3, 256, 256).to(device)
    out = Net(x1, x2)
    print([o.shape for o in out])
    # cp1 = res(x1)
    # cp2 = res(x2)