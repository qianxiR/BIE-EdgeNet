# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from typing import List, Optional, Tuple, Union
# from mmseg.registry import MODELS
# from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmcv.cnn import ConvModule
# from mmseg.models.backbones.resnet import BasicBlock
# from timm.models.swin_transformer_v2 import PatchMerging, SwinTransformerV2Block
# import torch.utils.checkpoint as checkpoint
# from timm.layers import to_2tuple
# from mmseg.structures import SegDataSample
#
# _int_or_tuple_2_t = Union[int, Tuple[int, int]]
#
#
# class SwinTV2Block(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             out_dim: int,
#             input_resolution: _int_or_tuple_2_t,
#             depth: int = 2,
#             num_heads: int = 8,
#             window_size: _int_or_tuple_2_t = 8,
#             downsample: bool = False,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             drop_path: float = 0.,
#             norm_layer: nn.Module = nn.LayerNorm,
#             pretrained_window_size: _int_or_tuple_2_t = 0,
#             output_nchw: bool = False,
#     ) -> None:
#         super().__init__()
#         # self.dim = dim
#         # self.input_resolution = input_resolution
#         # self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
#         self.dim = dim
#         # 兼容单个 int 或 tuple 输入，例如 64 或 (64,64)
#         if isinstance(input_resolution, int):
#             input_resolution = (input_resolution, input_resolution)
#         self.input_resolution = input_resolution
#         self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
#
#         self.depth = depth
#         self.output_nchw = output_nchw
#         self.grad_checkpointing = False
#         window_size = to_2tuple(window_size)
#         shift_size = tuple([w // 2 for w in window_size])
#         if downsample:
#             self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
#         else:
#             assert dim == out_dim
#             self.downsample = nn.Identity()
#         self.blocks = nn.ModuleList([
#             SwinTransformerV2Block(
#                 dim=out_dim,
#                 input_resolution=self.output_resolution,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else shift_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 proj_drop=proj_drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer,
#                 pretrained_window_size=pretrained_window_size,
#             )
#             for i in range(depth)])
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.downsample(x)
#         for blk in self.blocks:
#             if self.grad_checkpointing and not torch.jit.is_scripting():
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         return x.permute(0, 3, 1, 2)
#
#     def _init_respostnorm(self) -> None:
#         for blk in self.blocks:
#             nn.init.constant_(blk.norm1.bias, 0)
#             nn.init.constant_(blk.norm1.weight, 0)
#             nn.init.constant_(blk.norm2.bias, 0)
#             nn.init.constant_(blk.norm2.weight, 0)
#
#
# class SELayerIn2Out(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_cfg):
#         super().__init__()
#         self.conv1 = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv2 = ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
#         self.conv3 = ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         residual = self.conv1(x)
#         out = self.avg_pool(residual)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.sigmoid(out)
#         return residual * out
#
#
# class CDWeights(nn.Module):
#     def __init__(self, channels=128, norm_cfg=dict(type='BN', requires_grad=True)):
#         super(CDWeights, self).__init__()
#         self.convA = BasicBlock(channels, planes=channels, norm_cfg=norm_cfg)
#         self.convB = BasicBlock(channels, planes=channels, norm_cfg=norm_cfg)
#         self.sigmoid = nn.Sigmoid()
#
#     def spatial_difference(self, xA, xB):
#         xA_flat = xA.permute(0, 2, 3, 1).reshape(-1, xA.size(1))
#         xB_flat = xB.permute(0, 2, 3, 1).reshape(-1, xB.size(1))
#         cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)
#         cosine_sim = cosine_sim.view(xA.size(0), xA.size(2), xA.size(3))
#         cosine_sim = cosine_sim.unsqueeze(1)
#         c_weights = 1 - self.sigmoid(cosine_sim)
#         return c_weights
#
#     def channel_difference(self, xA, xB):
#         N, C, H, W = xA.shape
#         xA_flat = xA.view(N, C, -1)
#         xB_flat = xB.view(N, C, -1)
#         cosine_sim = 1 - self.sigmoid(F.cosine_similarity(xA_flat, xB_flat, dim=2))
#         hw_weights = cosine_sim.unsqueeze(-1).unsqueeze(-1)
#         return hw_weights
#
#     def forward(self, xA, xB):
#         c_weights = self.spatial_difference(xA, xB)
#         hw_weights = self.channel_difference(xA, xB)
#         c_weights_expanded = c_weights.expand(-1, hw_weights.size(1), -1, -1)
#         combined_weights = c_weights_expanded * hw_weights
#         xA_weighted = xA * combined_weights
#         xB_weighted = xB * combined_weights
#         xA_d = self.convA(xA_weighted)
#         xB_d = self.convB(xB_weighted)
#         outA = xA_d + xA
#         outB = xB_d + xB
#         return outA, outB
#
# @MODELS.register_module()
# class LENetEncoder(nn.Module):
#     def __init__(self):
#         super(LENetEncoder, self).__init__()
#         # self.model = timm.create_model('swinv2_base_window8_256', pretrained=True, pretrained_cfg_overlay=dict(
#         #     file='models/pretrained/swinv2_base_window8_256.ms_in1k/pytorch_model.bin'), features_only=True)
#         # self.model = timm.create_model(
#         #     'swinv2_base_window8_256',
#         #     pretrained=False,  # timm 自动下载 base 版本权重（embed_dim=128）
#         #     features_only=True,
#         #     embed_dim=128,
#         #     depths=[2, 2, 18, 2],  # SwinV2 Base 官方深度配置
#         #     num_heads=[4, 8, 16, 32],
#         # )
#         self.model = timm.create_model(
#             'swinv2_tiny_window8_256',
#             pretrained=False,
#             features_only=True,
#             embed_dim=64,
#             depths=[2, 2, 6, 2],
#             num_heads=[2, 4, 8, 16],
#         )
#
#
#         self.interaction_layers = ['blocks']
#         norm_cfg = dict(type='SyncBN', requires_grad=True)
#         # FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
#         # FPN_DICT = {'type': 'FPN', 'in_channels': [64, 128, 256, 512], 'out_channels': 256, 'num_outs': 4}
#         # self.fpnA = MODELS.build(FPN_DICT)
#         # self.fpnB = MODELS.build(FPN_DICT)
#         # self.decode_layersA = nn.Sequential(
#         #     nn.Identity(),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=64,
#         #                  num_heads=4),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=32,
#         #                  num_heads=8),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=16,
#         #                  num_heads=16),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=8,
#         #                  num_heads=32)
#         # )
#         # self.decode_layersB = nn.Sequential(
#         #     nn.Identity(),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=64,
#         #                  num_heads=4),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=32,
#         #                  num_heads=8),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=16,
#         #                  num_heads=16),
#         #     SwinTV2Block(dim=FPN_DICT['out_channels'], out_dim=FPN_DICT['out_channels'], input_resolution=8,
#         #                  num_heads=32)
#         # )
#         # self.channelA = nn.Sequential(
#         #     nn.Identity(),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg)
#         # )
#         # self.channelB = nn.Sequential(
#         #     nn.Identity(),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg),
#         #     SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2, out_channels=FPN_DICT['out_channels'],
#         #                   norm_cfg=norm_cfg)
#         # )
#         # self.sigmoid = nn.Sigmoid()
#         # self.re_weight3 = CDWeights(256, norm_cfg=norm_cfg)
#         # self.re_weight2 = CDWeights(256, norm_cfg=norm_cfg)
#         # self.re_weight1 = CDWeights(256, norm_cfg=norm_cfg)
#         # self.rwe = nn.Sequential(
#         #     CDWeights(64, norm_cfg=norm_cfg),  # 原 64→96
#         #     CDWeights(128, norm_cfg=norm_cfg),  # 原 128→192
#         #     CDWeights(256, norm_cfg=norm_cfg),  # 原 256→384
#         #     CDWeights(512, norm_cfg=norm_cfg),  # 原 512→768
#         #     CDWeights(1024, norm_cfg=norm_cfg)
#         # )
#         # --- 动态推断 backbone 特征通道并构建 FPN/RWE/Decode ---
#         # 尝试从 timm backbone 的 feature_info 推断 channels；若不存在则回退到默认
#         if hasattr(self.model, 'feature_info') and getattr(self.model, 'feature_info') is not None:
#             feat_info = self.model.feature_info.info
#             feat_channels = [f['num_chs'] for f in feat_info]  # e.g. [64,128,256,512]
#         else:
#             feat_channels = [64, 128, 256, 512]
#
#         # 用推断到的通道数构建 FPN 配置
#         FPN_DICT = {'type': 'FPN', 'in_channels': feat_channels, 'out_channels': 256, 'num_outs': len(feat_channels)}
#         self.fpnA = MODELS.build(FPN_DICT)
#         self.fpnB = MODELS.build(FPN_DICT)
#
#         # 计算每一级的空间分辨率（基于输入大小），默认输入 256，可通过 LENetEncoder(input_size=...) 调整
#         # 对 256x256，feat_levels 比如 stride=4,8,16,32 对应的 spatial: 64,32,16,8
#         input_size = 256
#         num_levels = len(feat_channels)
#         spatial_res = [input_size // (4 * (2 ** i)) for i in range(num_levels)]
#         spatial_res = [(r, r) for r in spatial_res]
#
#         # 为 decode_layers 构建 ModuleList（index 0 保留 Identity，后面为每级 SwinTV2Block）
#         default_heads = [4, 8, 16, 32, 64][:num_levels]
#         self.decode_layersA = nn.ModuleList([nn.Identity()] + [
#             SwinTV2Block(dim=FPN_DICT['out_channels'],
#                          out_dim=FPN_DICT['out_channels'],
#                          input_resolution=spatial_res[i],
#                          num_heads=default_heads[i])
#             for i in range(num_levels)
#         ])
#         self.decode_layersB = nn.ModuleList([nn.Identity()] + [
#             SwinTV2Block(dim=FPN_DICT['out_channels'],
#                          out_dim=FPN_DICT['out_channels'],
#                          input_resolution=spatial_res[i],
#                          num_heads=default_heads[i])
#             for i in range(num_levels)
#         ])
#
#         # channelA / channelB 也用 ModuleList，index0 为 Identity，后面对应每级
#         self.channelA = nn.ModuleList([nn.Identity()] + [
#             SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2,
#                           out_channels=FPN_DICT['out_channels'],
#                           norm_cfg=norm_cfg)
#             for _ in range(num_levels)
#         ])
#         self.channelB = nn.ModuleList([nn.Identity()] + [
#             SELayerIn2Out(in_channels=FPN_DICT['out_channels'] * 2,
#                           out_channels=FPN_DICT['out_channels'],
#                           norm_cfg=norm_cfg)
#             for _ in range(num_levels)
#         ])
#
#         # re-weight 模块，按 FPN 的 out_channels 构建（每一级一个）
#         self.re_weight_modules = nn.ModuleList([CDWeights(FPN_DICT['out_channels'], norm_cfg=norm_cfg)
#                                                 for _ in range(num_levels)])
#
#         # rwe 应该与 backbone 输出 levels 对应（使用 feat_channels）
#         self.rwe = nn.ModuleList([CDWeights(ch, norm_cfg=norm_cfg) for ch in feat_channels])
#
#
#     # def change_feature(self, x, y):
#     #     i = 2
#     #     for index in range(0, len(x), i):
#     #         x[index], y[index] = y[index], x[index]
#     #     return x, y
#     #
#     # def forward(self, xA, xB):
#     #     # xA = xA.permute(0, 2, 3, 1)  # [B,3,H,W] → [B,H,W,3]
#     #     # xB = xB.permute(0, 2, 3, 1)
#     #
#     #     xA_list = []
#     #     xB_list = []
#     #     ii = 0
#     #     for name, module in self.model.named_children():
#     #         if name in self.interaction_layers:
#     #             xA = module(xA)
#     #             xB = module(xB)
#     #         else:
#     #             xA = module(xA)
#     #             xB = module(xB)
#     #             xA = xA.permute(0, 3, 1, 2)
#     #             xB = xB.permute(0, 3, 1, 2)
#     #             xA, xB = self.rwe[ii](xA, xB)
#     #             xA_ = xA.permute(0, 2, 3, 1)
#     #             xB_ = xB.permute(0, 2, 3, 1)
#     #             xA_list.append(xA_)
#     #             xB_list.append(xB_)
#     #             ii += 1
#     #     xA_list = xA_list[1:]
#     #     xA_list = [x.permute(0, 3, 1, 2) for x in xA_list]
#     #     xB_list = xB_list[1:]
#     #     xB_list = [x.permute(0, 3, 1, 2) for x in xB_list]
#     #     xA_list, xB_list = self.change_feature(xA_list, xB_list)
#     #     xA_list = self.fpnA(xA_list)
#     #     xB_list = self.fpnB(xB_list)
#     #     xA_list, xB_list = self.change_feature(list(xA_list), list(xB_list))
#     #     xA_list = [x.permute(0, 2, 3, 1) for x in xA_list]
#     #     xB_list = [x.permute(0, 2, 3, 1) for x in xB_list]
#     #     xA1, xA2, xA3, xA4 = xA_list
#     #     xB1, xB2, xB3, xB4 = xB_list
#     #     change_maps = []
#     #     xA4_ = self.decode_layersA[4](xA4)
#     #     xB4_ = self.decode_layersB[4](xB4)
#     #     xA4_ = torch.cat([xA4_, xB4.permute(0, 3, 1, 2)], dim=1)
#     #     xB4_ = torch.cat([xB4_, xA4.permute(0, 3, 1, 2)], dim=1)
#     #     xA4 = self.channelA[4](xA4_)
#     #     xB4 = self.channelB[4](xB4_)
#     #     xA4 = F.interpolate(xA4, scale_factor=2, mode='bilinear', align_corners=False)
#     #     xB4 = F.interpolate(xB4, scale_factor=2, mode='bilinear', align_corners=False)
#     #     xA3 = xA3 + xB4.permute(0, 2, 3, 1)
#     #     xB3 = xB3 + xA4.permute(0, 2, 3, 1)
#     #     xA3_ = self.decode_layersA[3](xA3)
#     #     xB3_ = self.decode_layersB[3](xB3)
#     #     xA3_ = torch.cat([xA3_, xB3.permute(0, 3, 1, 2)], dim=1)
#     #     xB3_ = torch.cat([xB3_, xA3.permute(0, 3, 1, 2)], dim=1)
#     #     xA3 = self.channelA[3](xA3_)
#     #     xB3 = self.channelB[3](xB3_)
#     #     xA3 = F.interpolate(xA3, scale_factor=2, mode='bilinear', align_corners=False)
#     #     xB3 = F.interpolate(xB3, scale_factor=2, mode='bilinear', align_corners=False)
#     #     xA3, xB3 = self.re_weight3(xA3, xB3)
#     #     change_maps.append(torch.cat([xA3, xB3], dim=1))
#     #     xA2 = xA2 + xB3.permute(0, 2, 3, 1)
#     #     xB2 = xB2 + xA3.permute(0, 2, 3, 1)
#     #     xA2_ = self.decode_layersA[2](xA2)
#     #     xB2_ = self.decode_layersB[2](xB2)
#     #     xA2_ = torch.cat([xA2_, xB2.permute(0, 3, 1, 2)], dim=1)
#     #     xB2_ = torch.cat([xB2_, xA2.permute(0, 3, 1, 2)], dim=1)
#     #     xA2 = self.channelA[2](xA2_)
#     #     xB2 = self.channelB[2](xB2_)
#     #     xA2 = F.interpolate(xA2, scale_factor=2, mode='bilinear', align_corners=False)
#     #     xB2 = F.interpolate(xB2, scale_factor=2, mode='bilinear', align_corners=False)
#     #     xA2, xB2 = self.re_weight2(xA2, xB2)
#     #     change_maps.append(torch.cat([xA2, xB2], dim=1))
#     #     xA1 = xA1 + xB2.permute(0, 2, 3, 1)
#     #     xB1 = xB1 + xA2.permute(0, 2, 3, 1)
#     #     xA1_ = self.decode_layersA[1](xA1)
#     #     xB1_ = self.decode_layersB[1](xB1)
#     #     xA1_ = torch.cat([xA1_, xB1.permute(0, 3, 1, 2)], dim=1)
#     #     xB1_ = torch.cat([xB1_, xA1.permute(0, 3, 1, 2)], dim=1)
#     #     xA1 = self.channelA[1](xA1_)
#     #     xB1 = self.channelB[1](xB1_)
#     #     xA1, xB1 = self.re_weight1(xA1, xB1)
#     #     change_maps.append(torch.cat([xA1, xB1], dim=1))
#     #     return change_maps
#
#
#
#     # def change_feature(self, x, y):
#     #     i = 2
#     #     for index in range(0, len(x), i):
#     #         x[index], y[index] = y[index], x[index]
#     #     return x, y
#     #
#     # def forward(self, xA, xB):
#     #     # 1. 安全获取backbone多尺度特征（timm features_only=True返回NCHW列表：[B, C, H, W]）
#     #     if hasattr(self.model, 'forward_features'):
#     #         featsA = self.model.forward_features(xA)
#     #         featsB = self.model.forward_features(xB)
#     #     else:
#     #         featsA = self.model(xA)
#     #         featsB = self.model(xB)
#     #
#     #     # 2. 确保特征为NCHW格式（避免极少数NHWC情况）
#     #     def ensure_nchw(f):
#     #         if f.dim() == 4 and f.shape[1] < f.shape[-1]:  # 若通道维不是dim=1，转为NCHW
#     #             return f.permute(0, 3, 1, 2)
#     #         return f
#     #
#     #     featsA = [ensure_nchw(f) for f in featsA]
#     #     featsB = [ensure_nchw(f) for f in featsB]
#     #
#     #     # 3. 对每级特征应用rwe（re-weight），保持NCHW格式
#     #     xA_weighted = []
#     #     xB_weighted = []
#     #     for fa, fb, rwe_layer in zip(featsA, featsB, self.rwe):
#     #         fa_w, fb_w = rwe_layer(fa, fb)
#     #         xA_weighted.append(fa_w)
#     #         xB_weighted.append(fb_w)
#     #
#     #     # 4. FPN处理（输入输出均为NCHW列表）
#     #     xA_fpn = self.fpnA(xA_weighted)
#     #     xB_fpn = self.fpnB(xB_weighted)
#     #
#     #     # 5. 按原逻辑交换特征层
#     #     xA_fpn, xB_fpn = self.change_feature(list(xA_fpn), list(xB_fpn))
#     #
#     #     # 6. 转为NHWC格式供SwinTV2Block解码（Swin模块要求输入为NHWC）
#     #     xA_nhwc = [f.permute(0, 2, 3, 1) for f in xA_fpn]
#     #     xB_nhwc = [f.permute(0, 2, 3, 1) for f in xB_fpn]
#     #
#     #     # 7. 解码与特征融合（按特征层级动态索引，避免硬编码）
#     #     change_maps = []
#     #     num_levels = len(xA_nhwc)
#     #     # 从最深层（最小分辨率）到最浅层（最大分辨率）融合
#     #     for i in reversed(range(num_levels)):
#     #         # 当前层级特征
#     #         xa = xA_nhwc[i]
#     #         xb = xB_nhwc[i]
#     #
#     #         # 解码（使用ModuleList的正确索引，i+1对应decode_layers的层级）
#     #         xa_dec = self.decode_layersA[i + 1](xa)  # decode_layers[0]是Identity，跳过
#     #         xb_dec = self.decode_layersB[i + 1](xb)
#     #
#     #         # 通道注意力融合（确保输入为NCHW格式）
#     #         xa_cat = torch.cat([xa_dec, xb_dec.permute(0, 3, 1, 2)], dim=1)
#     #         xb_cat = torch.cat([xb_dec, xa_dec.permute(0, 3, 1, 2)], dim=1)
#     #         xa_att = self.channelA[i + 1](xa_cat)
#     #         xb_att = self.channelB[i + 1](xb_cat)
#     #
#     #         # 上采样融合到上一层（若不是最浅层）
#     #         if i > 0:
#     #             xa_up = F.interpolate(xa_att, scale_factor=2, mode='bilinear', align_corners=False)
#     #             xb_up = F.interpolate(xb_att, scale_factor=2, mode='bilinear', align_corners=False)
#     #             # 加到上一层特征（转为NCHW后相加）
#     #             xA_nhwc[i - 1] = xA_nhwc[i - 1].permute(0, 3, 1, 2) + xa_up
#     #             xB_nhwc[i - 1] = xB_nhwc[i - 1].permute(0, 3, 1, 2) + xb_up
#     #             # 转回NHWC供下一轮解码
#     #             xA_nhwc[i - 1] = xA_nhwc[i - 1].permute(0, 2, 3, 1)
#     #             xB_nhwc[i - 1] = xB_nhwc[i - 1].permute(0, 2, 3, 1)
#     #
#     #         # 应用re-weight并记录change map
#     #         xa_rw, xb_rw = self.re_weight_modules[i](xa_att, xb_att)
#     #         change_maps.append(torch.cat([xa_rw, xb_rw], dim=1))
#     #
#     #     # 保持原返回顺序（从浅到深）
#     #     change_maps = change_maps[::-1]
#     #     return change_maps
#
#     def change_feature(self, x, y):
#         i = 2
#         for index in range(0, len(x), i):
#             x[index], y[index] = y[index], x[index]
#         return x, y
#
#     def forward(self, xA, xB):
#         # 1. 安全获取backbone多尺度特征（timm features_only=True返回NCHW列表：[B, C, H, W]）
#         if hasattr(self.model, 'forward_features'):
#             featsA = self.model.forward_features(xA)
#             featsB = self.model.forward_features(xB)
#         else:
#             featsA = self.model(xA)
#             featsB = self.model(xB)
#
#         # 2. 确保特征为NCHW格式（避免极少数NHWC情况）
#         def ensure_nchw(f):
#             if f.dim() == 4 and f.shape[1] < f.shape[-1]:  # 通道维不是dim=1时转为NCHW
#                 return f.permute(0, 3, 1, 2)
#             return f
#
#         featsA = [ensure_nchw(f) for f in featsA]
#         featsB = [ensure_nchw(f) for f in featsB]
#
#         # 3. 对每级特征应用rwe（re-weight），保持NCHW格式
#         xA_weighted = []
#         xB_weighted = []
#         for fa, fb, rwe_layer in zip(featsA, featsB, self.rwe):
#             fa_w, fb_w = rwe_layer(fa, fb)
#             xA_weighted.append(fa_w)
#             xB_weighted.append(fb_w)
#
#         # 4. FPN处理（输入输出均为NCHW列表）
#         xA_fpn = self.fpnA(xA_weighted)
#         xB_fpn = self.fpnB(xB_weighted)
#
#         # 5. 按原逻辑交换特征层
#         xA_fpn, xB_fpn = self.change_feature(list(xA_fpn), list(xB_fpn))
#
#         # 6. 转为NHWC格式供SwinTV2Block解码（Swin模块要求输入为NHWC）
#         xA_nhwc = [f.permute(0, 2, 3, 1) for f in xA_fpn]  # [B, H, W, C]
#         xB_nhwc = [f.permute(0, 2, 3, 1) for f in xB_fpn]  # [B, H, W, C]
#
#         # 7. 解码与特征融合（严格保持维度一致性）
#         change_maps = []
#         num_levels = len(xA_nhwc)  # 通常为4（对应4级特征）
#
#         # 从最深层（最小分辨率）到最浅层（最大分辨率）融合
#         for i in reversed(range(num_levels)):
#             # 当前层级特征（NHWC格式，输入SwinTV2Block）
#             xa_nhwc = xA_nhwc[i]  # [B, H_i, W_i, C]
#             xb_nhwc = xB_nhwc[i]  # [B, H_i, W_i, C]
#
#             # 解码：SwinTV2Block输入NHWC，输出NCHW（[B, C, H_i, W_i]）
#             xa_dec = self.decode_layersA[i + 1](xa_nhwc)  # index0是Identity，跳过
#             xb_dec = self.decode_layersB[i + 1](xb_nhwc)
#
#             # 关键修复：直接拼接NCHW格式的特征（无需额外permute）
#             # xa_dec/xb_dec都是 [B, 256, H_i, W_i]，拼接后为 [B, 512, H_i, W_i]
#             xa_cat = torch.cat([xa_dec, xb_dec], dim=1)
#             xb_cat = torch.cat([xb_dec, xa_dec], dim=1)
#
#             # 通道注意力加权（输入输出均为NCHW）
#             xa_att = self.channelA[i + 1](xa_cat)  # [B, 256, H_i, W_i]
#             xb_att = self.channelB[i + 1](xb_cat)  # [B, 256, H_i, W_i]
#
#             # 上采样融合到上一层（若不是最浅层）
#             if i > 0:
#                 # 上采样当前层到上一层的分辨率（H_{i-1}=2*H_i, W_{i-1}=2*W_i）
#                 xa_up = F.interpolate(xa_att, scale_factor=2, mode='bilinear', align_corners=False)
#                 xb_up = F.interpolate(xb_att, scale_factor=2, mode='bilinear', align_corners=False)
#
#                 # 上一层特征转为NCHW后相加（原格式NHWC）
#                 xa_prev_nchw = xA_nhwc[i - 1].permute(0, 3, 1, 2)  # [B, 256, H_{i-1}, W_{i-1}]
#                 xb_prev_nchw = xB_nhwc[i - 1].permute(0, 3, 1, 2)  # [B, 256, H_{i-1}, W_{i-1}]
#
#                 # 融合后转回NHWC，供下一轮解码
#                 xA_nhwc[i - 1] = (xa_prev_nchw + xa_up).permute(0, 2, 3, 1)
#                 xB_nhwc[i - 1] = (xb_prev_nchw + xb_up).permute(0, 2, 3, 1)
#
#             # 应用re-weight模块，拼接生成change map
#             xa_rw, xb_rw = self.re_weight_modules[i](xa_att, xb_att)
#             change_map = torch.cat([xa_rw, xb_rw], dim=1)  # [B, 512, H_i, W_i]
#             change_maps.append(change_map)
#
#         # 保持原返回顺序（从浅到深：大分辨率→小分辨率，对应in_index=0,1,2）
#         change_maps = change_maps[::-1]
#         return change_maps
#
#
# @MODELS.register_module()
# class LENetFCNHead(BaseDecodeHead):
#     def __init__(self,
#                  num_convs=2,
#                  kernel_size=3,
#                  concat_input=True,
#                  dilation=1,
#                  **kwargs):
#         assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
#         self.num_convs = num_convs
#         self.concat_input = concat_input
#         self.kernel_size = kernel_size
#         super().__init__(**kwargs)
#         if num_convs == 0:
#             assert self.in_channels == self.channels
#         conv_padding = (kernel_size // 2) * dilation
#         convs = []
#         convs.append(
#             ConvModule(
#                 self.in_channels,
#                 self.channels,
#                 kernel_size=kernel_size,
#                 padding=conv_padding,
#                 dilation=dilation,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg))
#         for i in range(num_convs - 1):
#             convs.append(
#                 ConvModule(
#                     self.channels,
#                     self.channels,
#                     kernel_size=kernel_size,
#                     padding=conv_padding,
#                     dilation=dilation,
#                     conv_cfg=self.conv_cfg,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))
#         if num_convs == 0:
#             self.convs = nn.Identity()
#         else:
#             self.convs = nn.Sequential(*convs)
#         if self.concat_input:
#             self.conv_cat = ConvModule(
#                 self.in_channels + self.channels,
#                 self.channels,
#                 kernel_size=kernel_size,
#                 padding=kernel_size // 2,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg)
#
#     def _forward_feature(self, inputs):
#         x = self._transform_inputs(inputs)
#         feats = self.convs(x)
#         if self.concat_input:
#             feats = self.conv_cat(torch.cat([x, feats], dim=1))
#         return feats
#
#     def forward(self, inputs):
#         output = self._forward_feature(inputs)
#         output = self.cls_seg(output)
#         return output
#
#
# @MODELS.register_module()
# class MM_LENet4(EncoderDecoder):
#     def __init__(self, num_classes=2, norm_cfg=dict(type='SyncBN', requires_grad=True), **kwargs):
#         # 关键修复：backbone 改为配置字典（旧版本要求）
#         backbone = dict(type='LENetEncoder')
#         # decode_head 保持配置字典格式（无需改动）
#         decode_head = dict(
#             type='LENetFCNHead',
#             in_channels=512,
#             in_index=2,
#             channels=256,
#             num_convs=1,
#             concat_input=False,
#             dropout_ratio=0.1,
#             num_classes=num_classes,
#             norm_cfg=norm_cfg,
#             align_corners=False
#         )
#         # auxiliary_head 改为配置字典列表（旧版本要求）
#         auxiliary_head = [
#             dict(
#                 type='LENetFCNHead',
#                 in_channels=512,
#                 in_index=0,
#                 channels=256,
#                 num_convs=1,
#                 concat_input=False,
#                 dropout_ratio=0.1,
#                 num_classes=num_classes,
#                 norm_cfg=norm_cfg,
#                 align_corners=False
#             ),
#             dict(
#                 type='LENetFCNHead',
#                 in_channels=512,
#                 in_index=1,
#                 channels=256,
#                 num_convs=1,
#                 concat_input=False,
#                 dropout_ratio=0.1,
#                 num_classes=num_classes,
#                 norm_cfg=norm_cfg,
#                 align_corners=False
#             )
#         ]
#
#         # 按旧版本规范初始化父类（所有核心模块都传配置字典）
#         super().__init__(
#             backbone=backbone,
#             decode_head=decode_head,
#             auxiliary_head=auxiliary_head,
#             **kwargs
#         )
#
#         # 保留 self.encoder 引用（供 extract_feat 使用，父类已自动构建 backbone 实例）
#         self.encoder = self.backbone  # self.backbone 是父类通过配置字典构建的 LENetEncoder 实例
#         self.num_classes = num_classes
#         self.norm_cfg = norm_cfg
#
#     # extract_feat 和 forward 方法完全不变！
#     def extract_feat(self, x):
#         xA = x[:, 0:3, :, :]
#         xB = x[:, 3:6, :, :]
#         # xA = xA.permute(0, 2, 3, 1)
#         # xB = xB.permute(0, 2, 3, 1)
#         feats = self.encoder(xA, xB)
#         return feats
#
#     def forward(self, inputs, data_samples=None, mode='tensor'):
#         x = self.extract_feat(inputs)
#         main_logits = self.decode_head(x)
#         aux_logits = [head(x) for head in self.auxiliary_head]
#
#         if mode == 'predict':
#             main_logits = F.interpolate(main_logits, size=inputs.shape[2:], mode='bilinear', align_corners=False)
#             return self.post_process_result(main_logits, data_samples)
#         elif mode == 'tensor':
#             return main_logits, aux_logits
#         return main_logits, aux_logits