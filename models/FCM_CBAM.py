import torch
import torch.nn as nn
from models.CBAM import CBAMBlock


def calculate_padding(kernel_size, padding=None, dilation=1):
    """计算保持相同输出尺寸所需的填充量"""
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [
            dilation * (x - 1) + 1 for x in kernel_size
        ]
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
    return padding


class ConvolutionLayer(nn.Module):
    """标准卷积层，包含卷积、批归一化和激活函数"""
    default_activation = nn.Hardswish()

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            calculate_padding(kernel_size, padding, dilation),
            groups=groups, dilation=dilation, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = self.default_activation if activation is True else \
            activation if isinstance(activation, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))

    def forward_fused(self, x):
        return self.activation(self.conv(x))


# 标准CBAM模块（适配你的FCM，轻量化版本）
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力：平均+最大双池化 + 轻量MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.Hardswish(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # 空间注意力：3×3卷积捕捉局部空间依赖
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Hardswish()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x


class FCM(nn.Module):
    """特征组合模块-遥感变化检测优化版（替换为CBAM）"""

    def __init__(self, channels):
        super().__init__()
        self.main_channels = channels - channels // 4
        self.sub_channels = channels // 4

        # 主分支处理大部分通道（保留你的原逻辑）
        self.main_branch_conv1 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3, stride=1,
                                                  padding=1)
        self.main_branch_conv2 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3, stride=1,
                                                  padding=1)
        self.main_branch_conv3 = ConvolutionLayer(self.main_channels, channels, kernel_size=1, stride=1)

        # 子分支处理剩余通道（保留你的原逻辑）
        self.sub_branch_conv = ConvolutionLayer(self.sub_channels, channels, kernel_size=1, stride=1)

        # ✅ 核心替换：自定义注意力 → CBAM（轻量化版本，适配遥感）
        self.cbam = CBAMBlock(channels, reduction=16)  # reduction=16平衡计算量和效果

        # ✅ 保留你的核心优化：可学习缩放/融合权重
        self.spatial_scale = nn.Parameter(torch.tensor(1.0))
        self.channel_scale = nn.Parameter(torch.tensor(1.0))
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """特征组合模块前向传播"""
        # 特征分割（保留原逻辑）
        main_features, sub_features = torch.split(x, [self.main_channels, self.sub_channels], dim=1)

        # 主分支特征处理（保留原逻辑）
        processed_main_features = self.main_branch_conv1(main_features)
        processed_main_features = self.main_branch_conv2(processed_main_features)
        processed_main_features = self.main_branch_conv3(processed_main_features)

        # 子分支特征处理（保留原逻辑）
        processed_sub_features = self.sub_branch_conv(sub_features)

        # ✅ 替换注意力逻辑：CBAM一次性完成通道+空间注意力
        # 先融合主/子分支特征，再用CBAM增强
        fused_features = self.fusion_weight * processed_main_features + (
                    1 - self.fusion_weight) * processed_sub_features
        cbam_enhanced = self.cbam(fused_features)

        # ✅ 保留你的可学习缩放（适配变化检测的注意力强度）
        final_features = self.channel_scale * self.spatial_scale * cbam_enhanced + fused_features  # 残差连接更稳定

        return final_features


if __name__ == "__main__":
    input = torch.randn(1, 256, 32, 32)  # 模拟3阶段特征
    fcm = FCM(256)
    output = fcm(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")  # 输出仍为[1,256,32,32]，维度完全匹配