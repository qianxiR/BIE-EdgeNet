import torch
import torch.nn as nn

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