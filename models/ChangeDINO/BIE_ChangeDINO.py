import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

from models.ChangeDINO.blocks.fpn import FPN, DsBnRelu
from models.ChangeDINO.blocks.cbam import CBAM1
from models.CBAM import CBAMBlock
from models.ChangeDINO.blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from models.ChangeDINO.blocks.diffatts import TransformerBlock
from models.ChangeDINO.blocks.refine import LearnableSoftMorph
from models.ChangeDINO.mobilenetv2 import mobilenet_v2

from models.submodules import *

from einops import rearrange
from models.BIE_Cross_Attentions import *

class Pre_Post_TemporalBIE(nn.Module):
    """双向跨时相BIE模块：输出两个增强特征（f2增强f1 + f1增强f2）"""

    def __init__(self, nf, heads, reduction, stage=None):
        super().__init__()
        self.att_stage = stage

        # 特征对齐：统一x1/x2特征分布（确保双时相特征可对比）
        self.align1 = nn.Conv2d(nf, nf, 1, bias=False)
        self.align2 = nn.Conv2d(nf, nf, 1, bias=False)

        # self.diff_enhancer = FourStage_Diff_Enhancer()

        # # 双时相注意力：基于f1、f2和差异生成权重（覆盖双向视角）
        # self.attn = nn.Sequential(
        #     nn.Conv2d(nf * 3, nf, 1),  # 输入：f1 + f2 + 差异特征（3*nf通道）
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(nf, nf, 1),
        #     nn.Sigmoid()  # 注意力权重（0~1，突出变化区域）
        # )
        # 替换为轻量化双向交叉注意力（核心优化）
        # self.attn = LightWeight_DifferenceFeatureComplementaryAttention(nf, heads=heads, reduction=reduction)
        self.attn = DW_DifferenceFeatureComplementaryAttention(nf, heads=heads, reduction=reduction)
        # self.attn = DiffAttention(nf, heads=heads, reduction=reduction)

        self.CBAM1 = CBAMBlock(nf)
        self.CBAM2 = CBAMBlock(nf)

        # 单独归一化：分别稳定两个增强特征的分布
        self.norm1 = LayerNorm2d(nf)  # 用于enhanced_f1
        self.norm2 = LayerNorm2d(nf)  # 用于enhanced_f2


        # 权重初始化
        self.initialize_weights(self.attn, 0.1)
        # self.initialize_weights(self.CBAM1, 0.1)
        # self.initialize_weights(self.CBAM2, 0.1)
        # self.initialize_weights(self.norm1, 0.1)
        # self.initialize_weights(self.norm2, 0.1)
        nn.init.normal_(self.align1.weight, mean=0, std=0.01)
        nn.init.normal_(self.align2.weight, mean=0, std=0.01)
        self.sigmoid = nn.Sigmoid()

        # self.alpha = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def initialize_weights(net_l, scale=0.1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)

    # def spatial_difference(self, xA, xB):
    #     xA_flat = xA.permute(0, 2, 3, 1).reshape(-1, xA.size(1))
    #     xB_flat = xB.permute(0, 2, 3, 1).reshape(-1, xB.size(1))
    #     cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)
    #     cosine_sim = cosine_sim.view(xA.size(0), xA.size(2), xA.size(3))
    #     cosine_sim = cosine_sim.unsqueeze(1)
    #     c_weights = 1-self.sigmoid(cosine_sim)
    #     return c_weights

    def forward(self, feat_x1, feat_x2):
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

        # 计算空间权重和通道权重
        c_weights = self.spatial_difference(f1, f2)  # (2, 1, 32, 32)
        hw_weights = self.channel_difference(f1, f2)  # (2, 128, 1, 1)

        # 权重归一化，避免数值爆炸
        c_weights_max = c_weights.max(dim=3, keepdim=True)[0]  # 先计算宽度维度max
        c_weights_max = c_weights_max.max(dim=2, keepdim=True)[0]  # 再计算高度维度max
        c_weights = c_weights / (c_weights_max + 1e-6)  # 空间权重归一化


        hw_weights_max = hw_weights.max(dim=1, keepdim=True)[0]
        hw_weights = hw_weights / (hw_weights_max + 1e-6)

        # 将 c_weights 扩展到与 hw_weights 相同的形状
        c_weights_expanded = c_weights.expand(-1, hw_weights.size(1), -1, -1)  # (2, 128, 32, 32)

        # 合并权重 (比如可以选择相乘，也可以进行加权平均)
        combined_weights = c_weights_expanded * hw_weights  # (2, 128, 32, 32)

        # # 对 xA 和 xB 进行加权处理
        # xA_weighted = f1 * combined_weights
        # xB_weighted = f2 * combined_weights
        #
        # # 2. 差异特征与双向注意力：捕捉双时相变化
        abs_diff = torch.abs(f1 - f2)  # 显式计算变化区域
        diff_feat = abs_diff * (1 + combined_weights)

        # diff_enhanced = self.diff_enhancer(diff_feat, stage=self.att_stage) + diff_feat

        # attn_weight = self.attn(torch.cat([f1, f2, diff_feat], dim=1))  # 融合双时相视角生成权重
        #
        # # 3. 双向增强：用对方特征强化自身变化区域
        # # enhanced_f1：x1的基础上，用x2的变化区域特征增强
        # enhanced_f1 = f1 + attn_weight * f2
        # # enhanced_f2：x2的基础上，用x1的变化区域特征增强
        # enhanced_f2 = f2 + attn_weight * f1

        # 4. 轻量化双向交叉注意力（输出两个方向的权重）
        attn_weight_f1, attn_weight_f2= self.attn(f1, f2, diff_feat)

        # 5. 双向增强
        # gate1 = self.sigmoid(attn_weight_f1) * self.alpha
        # gate2 = self.sigmoid(attn_weight_f2) * self.alpha
        gate1 = self.sigmoid(attn_weight_f1 * combined_weights) + 1e-3
        gate2 = self.sigmoid(attn_weight_f2 * combined_weights) + 1e-3
        enhanced_f1 = f1 + gate1 * attn_weight_f1 * f2
        enhanced_f2 = f2 + gate2 * attn_weight_f2 * f1
        # enhanced_f1 = f1 + attn_weight_f1 * f2
        # enhanced_f2 = f2 + attn_weight_f2 * f1

        # enhanced_f1 = self.CBAM1(enhanced_f1)
        # enhanced_f2 = self.CBAM2(enhanced_f2)

        # 4. 残差归一化：保留各自原始特征，稳定梯度
        # 对enhanced_f1：残差连接x1的原始特征（确保x1基础信息不丢失）
        enhanced_f1 = self.norm1(enhanced_f1 + feat_x1)
        # 对enhanced_f2：残差连接x2的原始特征（确保x2基础信息不丢失）
        enhanced_f2 = self.norm2(enhanced_f2 + feat_x2)


        enhanced_f1 = self.CBAM1(enhanced_f1)
        enhanced_f2 = self.CBAM2(enhanced_f2)

        return enhanced_f1, enhanced_f2, combined_weights # 输出两个增强特征

    # def spatial_difference(self, xA, xB):
    #     B, C, H, W = xA.shape
    #     # 保留空间维度：(B, C, H, W) → (B, C, H*W)，每个位置对应一个特征向量
    #     xA_flat = xA.view(B, C, H*W)  # (B, C, HW)
    #     xB_flat = xB.view(B, C, H*W)  # (B, C, HW)
    #     # 按通道维度计算每个空间位置的余弦相似度（dim=1）
    #     cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)  # (B, HW)
    #     # 恢复空间维度：(B, 1, H, W)，与输入特征同尺寸
    #     cosine_sim = cosine_sim.view(B, 1, H, W)
    #     c_weights = 1 - self.sigmoid(cosine_sim)  # 差异越大，权重越高
    #     return c_weights
    #
    # def channel_difference(self, xA, xB):
    #     N, C, H, W = xA.shape
    #     xA_flat = xA.view(N, C, -1)
    #     xB_flat = xB.view(N, C, -1)
    #     cosine_sim = 1-self.sigmoid(F.cosine_similarity(xA_flat, xB_flat, dim=2))
    #     hw_weights = cosine_sim.unsqueeze(-1).unsqueeze(-1)
    #     return hw_weights

    def spatial_difference(self, xA, xB):
        """修正：计算每个空间位置的特征差异，放大差距"""
        B, C, H, W = xA.shape
        # 维度调整：(B, C, H, W) → (B, H*W, C)（每个空间位置对应一个特征向量）
        xA_flat = xA.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)
        xB_flat = xB.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)
        # 计算每个空间位置的特征相似度（dim=2：特征维度）
        cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=2)  # (B, HW)
        cosine_sim = cosine_sim.view(B, 1, H, W)
        # 修正：用指数函数放大差距
        c_weights = torch.exp((1 - self.sigmoid(cosine_sim)))
        return c_weights

    def channel_difference(self, xA, xB):
        """修正：计算每个通道的局部差异，放大差距"""
        N, C, H, W = xA.shape
        # 维度调整：(N, C, H, W) → (N, C, H*W)，保留通道维度，展平空间维度
        xA_flat = xA.view(N, C, H * W)  # (N, C, HW)
        xB_flat = xB.view(N, C, H * W)  # (N, C, HW)
        # 按通道计算空间维度的余弦相似度（dim=2），一次性完成所有通道计算
        cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=2)  # (N, C)
        # 放大差异+恢复维度：(N, C) → (N, C, 1, 1)，与输入特征广播兼容
        weight = torch.exp((1 - self.sigmoid(cosine_sim)))
        hw_weights = weight.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        return hw_weights


def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]

    elif backbone_name == "resnet34":
        backbone = timm.create_model("resnet34", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128, 128],
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]), CBAM1(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]), CBAM1(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]), CBAM1(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]), CBAM1(in_dims[0], 8)
        )

    def forward(self, feas, ds_feas):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_feas  # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4


class Encoder(nn.Module):
    def __init__(
        self,
        backbone="mobilenetv2",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",
        dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)
        self.dense_adp = DenseAdapterLite(
            in_dim=1024, out_dim=dense_out_dim, bottleneck=fpn_channels // 2
        )
        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """
        fea = self.backbone.forward(x)
        fea = self.fpn(fea[-4:])  # t1_p1, t1_p2, t1_p3, t1_p4

        ds_fea = self.dino(x)  # [B, N, C]

        # process dense features
        ds_fea = self.dense_adp(ds_fea)

        fea = self.pff(fea, ds_fea)

        return fea


class FuseGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1, bias=True), 
            nn.Sigmoid()
        )
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)


# Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

class Detector(nn.Module):
    def __init__(
        self,
        fpn_channels=128,
        n_layers=[1, 1 ,1, 1],
        **kwargs,
    ):
        super().__init__()
        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)


        self.diff_conv2 = conv_diff(fpn_channels*3, fpn_channels)
        self.diff_conv3 = conv_diff(fpn_channels*3, fpn_channels)
        self.diff_conv4 = conv_diff(fpn_channels*3, fpn_channels)
        self.diff_conv5 = conv_diff(fpn_channels*3, fpn_channels)

        self.tb5 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="CDA",
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=3,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",)
            for _ in range(n_layers[0])]
        )
        self.tb4 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="CDA",
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=3,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",)
            for _ in range(n_layers[1])]
        )
        self.tb3 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="OCDA",
                window_size=8,
                overlap_ratio=0.5,
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=2,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            )
            for _ in range(n_layers[2])]
        )
        self.tb2 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="OCDA",
                window_size=8,
                overlap_ratio=0.5,
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=1,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            )
            for _ in range(n_layers[3])]
        )
        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)

    def forward(self, x1s, x2s, cos_weights):
        ### Extract backbone features
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        cos_p2, cos_p3, cos_p4, cos_p5 = cos_weights


        # diff_p2 = torch.abs(t1_p2 - t2_p2)
        # diff_p3 = torch.abs(t1_p3 - t2_p3)
        # diff_p4 = torch.abs(t1_p4 - t2_p4)
        # diff_p5 = torch.abs(t1_p5 - t2_p5)

        
        diff_p2 = self.diff_conv2(torch.cat([
            t1_p2 * cos_p2,
            t2_p2 * cos_p2,
            torch.abs(t1_p2 - t2_p2)
        ], dim=1))
        diff_p3 = self.diff_conv3(torch.cat([
            t1_p3 * cos_p3,
            t2_p3 * cos_p3,
            torch.abs(t1_p3 - t2_p3)
        ], dim=1))
        diff_p4 = self.diff_conv4(torch.cat([
            t1_p4 * cos_p4,
            t2_p4 * cos_p4,
            torch.abs(t1_p4 - t2_p4)
        ], dim=1))
        diff_p5 = self.diff_conv5(torch.cat([
            t1_p5 * cos_p5,
            t2_p5 * cos_p5,
            torch.abs(t1_p5 - t2_p5)
        ], dim=1))



        fea_p5 = self.tb5(diff_p5)
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        fea_p2 = self.tb2(fea_p2)
        pred_p2 = self.p2_head(fea_p2)

        pred_p2 = F.interpolate(
            pred_p2, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p5 = F.interpolate(
            pred_p5, size=(256, 256), mode="bilinear", align_corners=False
        )

        return pred_p2, pred_p3, pred_p4, pred_p5


class BIE_ChangeDINO(nn.Module):
    def __init__(self, backbone="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1, 1], **kwargs):
        super().__init__()
        self. encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)

        self.temporal_bie1 = Pre_Post_TemporalBIE(nf=fpn_channels, heads=4, reduction=2)  
        self.temporal_bie2 = Pre_Post_TemporalBIE(nf=fpn_channels, heads=4, reduction=2)  
        self.temporal_bie3 = Pre_Post_TemporalBIE(nf=fpn_channels, heads=4, reduction=2)  
        self.temporal_bie4 = Pre_Post_TemporalBIE(nf=fpn_channels, heads=4, reduction=2)  

        self.detector = Detector(fpn_channels=fpn_channels, n_layers=n_layers,**kwargs)
        self.refiner = LearnableSoftMorph(3, 5)

    @torch.inference_mode()
    def _forward(self, x1, x2):
        # for inference
        fea1 = self.encoder(x1)
        fea2 = self.encoder(x2)

         # BIE增强
        enhanced_x1_fa1, enhanced_x2_fa1, diff_feat_1 = self.temporal_bie1(fea1[0], fea2[0])
        enhanced_x1_fa2, enhanced_x2_fa2, diff_feat_2 = self.temporal_bie2(fea1[1], fea2[1])
        enhanced_x1_fa3, enhanced_x2_fa3, diff_feat_3 = self.temporal_bie3(fea1[2], fea2[2])
        enhanced_x1_fa4, enhanced_x2_fa4, diff_feat_4 = self.temporal_bie4(fea1[3], fea2[3])
        
        x1_fa_enhanced = [enhanced_x1_fa1, enhanced_x1_fa2, enhanced_x1_fa3, enhanced_x1_fa4]
        x2_fa_enhanced = [enhanced_x2_fa1, enhanced_x2_fa2, enhanced_x2_fa3, enhanced_x2_fa4]
        diff_feat_stage = [diff_feat_1, diff_feat_2, diff_feat_3, diff_feat_4]
        
        preds = self.detector(x1_fa_enhanced, x2_fa_enhanced, diff_feat_stage)

        # pred, _, _, _ = self.detector(fea1, fea2)
        pred = self.refiner(preds[0])
        return pred

    def forward(self, x1, x2):
        # for training
        ## change detection
        fea1 = self.encoder(x1)
        fea2 = self.encoder(x2)

        enhanced_x1_fa1, enhanced_x2_fa1, diff_feat_1 = self.temporal_bie1(fea1[0], fea2[0])  # Stage1
        enhanced_x1_fa2, enhanced_x2_fa2, diff_feat_2 = self.temporal_bie2(fea1[1], fea2[1])  # Stage2
        enhanced_x1_fa3, enhanced_x2_fa3, diff_feat_3 = self.temporal_bie3(fea1[2], fea2[2])  # Stage3
        enhanced_x1_fa4, enhanced_x2_fa4, diff_feat_4 = self.temporal_bie4(fea1[3], fea2[3])  # Stage4

        # 增强后的FA特征列表
        x1_fa_enhanced = [enhanced_x1_fa1, enhanced_x1_fa2, enhanced_x1_fa3, enhanced_x1_fa4]
        x2_fa_enhanced = [enhanced_x2_fa1, enhanced_x2_fa2, enhanced_x2_fa3, enhanced_x2_fa4]
        diff_feat_stage = [diff_feat_1, diff_feat_2, diff_feat_3, diff_feat_4]

        preds = self.detector(x1_fa_enhanced, x2_fa_enhanced, diff_feat_stage)
        final_pred = self.refiner(preds[0])
        return final_pred, preds  # pred, pred_p2, pred_p3, pred_p4, pred_p5

if __name__ == '__main__':
    # res = InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=1, skip_connection=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = BIE_ChangeDINO(
            backbone="mobilenetv2",
            # backbone="resnet34",
            fpn_name="fpn",
            fpn_channels=128,
            deform_groups=4,
            gamma_mode="SE",
            beta_mode="contextgatedconv",
            n_layers=[1, 1, 1, 1],
            extract_ids=[5, 11, 17, 23],
        ).to(device)
    x1 = torch.randn(16, 3, 256, 256).to(device)
    x2 = torch.randn(16, 3, 256, 256).to(device)
    out = Net(x1, x2)
    print(out[0].shape)