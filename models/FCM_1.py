import torch
import torch.nn as nn


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
    # ✅ 优化1：激活函数替换为Hardswish，强化遥感特征的对比度，梯度更稳定
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


class ChannelAttention(nn.Module):
    """通道注意力模块，通过卷积和全局池化提取通道特征"""

    def __init__(self, channels):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=3,
            stride=1, padding=1, groups=channels
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.global_pooling(x)
        attention_map = self.sigmoid(x)
        return attention_map


class SpatialAttention(nn.Module):
    """空间注意力模块，通过卷积提取空间特征"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        attention_map = self.sigmoid(x)
        return attention_map


class FCM(nn.Module):
    """特征组合模块-遥感变化检测优化版，结合通道和空间注意力融合多尺度特征"""

    def __init__(self, channels):
        super().__init__()
        self.main_channels = channels - channels // 4
        self.sub_channels = channels // 4

        # 主分支处理大部分通道
        self.main_branch_conv1 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3, stride=1,
                                                  padding=1)
        self.main_branch_conv2 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3, stride=1,
                                                  padding=1)
        self.main_branch_conv3 = ConvolutionLayer(self.main_channels, channels, kernel_size=1, stride=1)

        # 子分支处理剩余通道
        self.sub_branch_conv = ConvolutionLayer(self.sub_channels, channels, kernel_size=1, stride=1)

        # 注意力模块
        self.spatial_attention = SpatialAttention(channels)
        self.channel_attention = ChannelAttention(channels)

        # ✅ 优化2：添加可学习的注意力缩放系数，自适应调整注意力强度（高维特征必备）
        self.spatial_scale = nn.Parameter(torch.tensor(1.0))
        self.channel_scale = nn.Parameter(torch.tensor(1.0))

        # ✅ 优化3：添加可学习的动态融合权重，自适应平衡主/子分支特征
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """特征组合模块前向传播"""
        # 特征分割
        main_features, sub_features = torch.split(x, [self.main_channels, self.sub_channels], dim=1)

        # 主分支特征处理
        processed_main_features = self.main_branch_conv1(main_features)
        processed_main_features = self.main_branch_conv2(processed_main_features)
        processed_main_features = self.main_branch_conv3(processed_main_features)

        # 子分支特征处理
        processed_sub_features = self.sub_branch_conv(sub_features)

        # 注意力机制应用 + 可学习缩放
        spatial_attended_features = self.spatial_scale * self.spatial_attention(
            processed_sub_features) * processed_main_features
        channel_attended_features = self.channel_scale * self.channel_attention(
            processed_main_features) * processed_sub_features

        # ✅ 优化3：动态加权融合（替代简单加法），兼顾语义和细节
        combined_features = self.fusion_weight * spatial_attended_features + (
                    1 - self.fusion_weight) * channel_attended_features

        return combined_features


if __name__ == "__main__":
    input = torch.randn(1, 256, 32, 32)  # 模拟你的3阶段特征
    FCM = FCM(256)
    output = FCM(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")