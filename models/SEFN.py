import torch
import torch.nn as nn
import torch.nn.functional as F

"""
GatedSpatialRepair
"""
class Light_Gated_SEFN(nn.Module):
    """
    为你的场景定制：FA门控对齐+BIE-EdgeNet+32维embed_dim+变化检测
    核心：轻量+门控+无冗余，与FA无缝衔接，不稀释Trans全局差异特征
    """

    def __init__(self, dim=32, kernel_size=3):
        super().__init__()
        self.dim = dim
        # 1. 轻量化CNN空间特征提取：1x1降维+深度卷积做局部邻域增强，无下采样/上采样
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim // 2,
                      bias=False),  # 深度卷积
            nn.Conv2d(dim // 2, dim, kernel_size=1, padding=0, bias=False),
        )

        # 2. 关键：新增自适应门控层，和FA的gate逻辑一致！学习空间特征的注入权重，杜绝CNN过载
        self.gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # 3. 最后的轻量化融合卷积，强化特征交互
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cnn_feat):
        """
        x: FA门控融合后的特征 (B, dim, H, W) 你的核心特征
        cnn_feat: CNN分支的原始特征 (B, dim, H, W) 复用，无需重提取
        """
        # 提取CNN的轻量化空间先验
        spatial_feat = self.spatial_conv(cnn_feat)
        # 自适应门控：学习空间特征的注入比例，0~1之间，完美匹配FA的权重逻辑
        gate_weight = self.gate(torch.cat([x, spatial_feat], dim=1))
        # 门控融合：空间特征 * 权重 + 原始特征，温和注入，不稀释Trans特征
        x_fuse = gate_weight * spatial_feat + (1 - gate_weight) * x
        # 轻量化卷积增强局部结构，修复Trans的空间退化
        x_out = self.dwconv(x_fuse)
        x_out = self.norm(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # 残差连接：保证梯度传导，防止特征丢失
        return x + x_out

if __name__ == '__main__':
    # 创建两个随机输入张量
    x1 = torch.randn(2, 32, 50, 50) # [batch, channels, height, width]
    x2 = torch.randn(2, 32, 50, 50)
    model = Light_Gated_SEFN(dim=32)
    output = model(x1, x2)
    print(f"输入张量形状: {x1.shape}")
    print(f"输入张量形状: {x2.shape}")
    print(f"输出张量形状: {output.shape}")