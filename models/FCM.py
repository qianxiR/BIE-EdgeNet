import torch
import torch.nn as nn

def calculate_padding(kernel_size, padding=None, dilation=1):
   """计算保持相同输出尺寸所需的填充量"""
   if dilation > 1:
       # 计算扩张后的实际卷积核大小
        kernel_size = dilation * (kernel_size - 1) + 1if isinstance(kernel_size, int) else [
            dilation * (x - 1) + 1for x in kernel_size
        ]
   if padding is None:
       # 自动计算填充量，使输出尺寸与输入相同
        padding = kernel_size // 2if isinstance(kernel_size, int) else [x // 2for x in kernel_size]
   return padding


class ConvolutionLayer(nn.Module):
   """标准卷积层，包含卷积、批归一化和激活函数"""
   default_activation = nn.SiLU() # 默认激活函数

   def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, activation=True):
       """初始化卷积层参数"""
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
       """前向传播：卷积 -> 批归一化 -> 激活"""
       return self.activation(self.batch_norm(self.conv(x)))

   def forward_fused(self, x):
       """融合批归一化的前向传播：卷积 -> 激活"""
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
       """通道注意力计算流程"""
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
       """空间注意力计算流程"""
       x = self.conv(x)
       x = self.batch_norm(x)
       attention_map = self.sigmoid(x)
       return attention_map


class FCM(nn.Module):
   """特征组合模块，结合通道和空间注意力融合多尺度特征"""
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

       # 注意力机制应用 ===> 可以更换我们之前的任意注意力模块
       spatial_attended_features = self.spatial_attention(processed_sub_features) * processed_main_features
       channel_attended_features = self.channel_attention(processed_main_features) * processed_sub_features

       # 特征融合
       combined_features = spatial_attended_features + channel_attended_features
       return combined_features

if __name__ == "__main__":
    input = torch.randn(1,32,50, 50)
    FCM = FCM(32)
    output = FCM(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")