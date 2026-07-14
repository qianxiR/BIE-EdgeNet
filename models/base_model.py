import torch.nn.functional
# from Demos.win32console_demo import window_size

from models.TransformerBaseNetworks import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from models.ChangeMambaBCD.vmamba import SS2D
from models.WaveFormer import Wave2D
from models.MHLA import MHLA_Normed_Torch as MHLA



from models.ChannelAggregationFFN import ChannelAggregationFFN

class ASPP_v1(nn.Module):
    def __init__(self, channel):
        super(ASPP_v1, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.leakyReLU = nn.LeakyReLU(inplace=True)
        self.conv_1x1_output = nn.Conv2d(channel * 4, channel, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.leakyReLU(self.dilate1(x))
        dilate2_out = self.leakyReLU(self.dilate2(x))
        dilate3_out = self.leakyReLU(self.dilate3(x))
        dilate4_out = self.leakyReLU(self.dilate4(x))
        out = self.conv_1x1_output(torch.cat([dilate1_out, dilate2_out,
                                              dilate3_out, dilate4_out], dim=1))
        return out

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x):
        # pdb.set_trace()
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Diff_SEM(nn.Module):
    def __init__(self, sem_channel, edge_channel, out_channel, stride=4):
        super(Diff_SEM, self).__init__()
        self.conv1 = nn.Conv2d(sem_channel, out_channel, 1, 1)
        self.conv2 = nn.Conv2d(edge_channel, out_channel, 1, 1)
        # self.up = nn.ConvTranspose2d(32, 1, kernel_size=stride, stride=stride)

        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        self.upconv2 = nn.Conv2d(out_channel, 1, 3, padding=1)
        self.probability = nn.Sigmoid()
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.LeakRelu = nn.LeakyReLU(inplace=True)

        self.diff_proj = nn.Conv2d(edge_channel, edge_channel, 1)  # 投影差异特征

    def forward(self, sem, edge, diff_feat=None):
        sem = self.conv1(sem)
        edge = self.conv2(edge)

        if diff_feat is not None:
            diff_feat_proj = self.diff_proj(diff_feat)  # 投影差异特征到边缘通道
            # 边缘特征与差异特征融合
            edge = edge + diff_feat_proj

        sem_up = self.upconv2(self.upsample(sem))
        sem_up = self.probability(sem_up)
        out = edge * sem_up
        out = self.conv3(out)
        out = self.bn(out)
        out = self.LeakRelu(out)
        return out

class SementicAddEdge(nn.Module):
    def __init__(self, sem_channel, edge_channel, out_channel, stride=2):
        super(SementicAddEdge, self).__init__()
        self.conv1 = nn.Conv2d(sem_channel, out_channel, 1, 1)
        self.conv2 = nn.Conv2d(edge_channel, out_channel, 1, 1)

        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        self.upconv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.LeakRelu = nn.LeakyReLU(inplace=True)

    def forward(self, sem, edge):
        sem = self.conv1(sem)
        edge = self.conv2(edge)
        sem_up = self.upconv2(self.upsample(sem))
        out = sem_up + edge
        out = self.conv3(out)
        out = self.bn(out)
        out = self.LeakRelu(out)
        return out


class DiffSementicAddEdge(nn.Module):
    def __init__(self, sem_channel, edge_channel, out_channel, stride=2, use_diff=True):
        super(DiffSementicAddEdge, self).__init__()
        self.use_diff = use_diff
        self.stride = stride


        self.conv1 = nn.Conv2d(sem_channel, out_channel, 1, 1)  # sem→out_channel
        self.conv2 = nn.Conv2d(edge_channel, out_channel, 1, 1)  # edge→out_channel

        # 新增：差异特征处理（通道对齐+权重生成）
        if self.use_diff:
            # diff_feat通道=edge_channel（与ESA模块一致，无需额外调整）
            self.diff_proj = nn.Conv2d(edge_channel, out_channel, 1, 1)  # 差异通道对齐
            self.diff_weight_conv = nn.Conv2d(out_channel, 1, 3, padding=1)  # 生成0-1权重
            self.sigmoid = nn.Sigmoid()


        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        self.upconv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)  # 上采样后语义细化


        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.LeakRelu = nn.LeakyReLU(inplace=True)

    def forward(self, sem, edge, diff_feat=None):
        sem = self.conv1(sem)  # (B, out_channel, H_sem, W_sem)
        edge = self.conv2(edge)  # (B, out_channel, H_edge, W_edge)

        # 2. 新增：差异特征处理（仅当启用且输入有效时）
        diff_weight = torch.ones(1, 1, edge.shape[2], edge.shape[3], device=sem.device)
        if self.use_diff and diff_feat is not None:
            diff_feat_proj = self.diff_proj(diff_feat)  # (B, out_channel, H_diff, W_diff)
            diff_feat_proj = F.interpolate(
                diff_feat_proj,
                size=edge.shape[2:],  # 对齐到edge的尺度（H_edge×W_edge）
                mode='bilinear',
                align_corners=False
            )

            # 2.3 生成差异引导权重（0-1：变化区≈1，非变化区≈0）
            diff_weight = self.diff_weight_conv(diff_feat_proj)
            diff_weight = self.sigmoid(diff_weight)  # (B, 1, H_edge, W_edge)

        # 3. 原有逻辑：语义上采样（对齐到edge尺度，适配你的stride）
        sem_up = self.upconv2(self.upsample(sem))  # (B, out_channel, H_edge, W_edge)

        # 4. 新增：差异引导的语义-边缘融合（核心创新）
        out = sem_up + edge  # 原有互补融合
        out = out * diff_weight  # 差异权重调制：聚焦真变化区

        # 5. 原有逻辑：特征精炼+激活
        out = self.conv3(out)
        out = self.bn(out)
        out = self.LeakRelu(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

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

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, res=14, feature_size=64):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sr_ratio = sr_ratio
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # self.attn = SS2D(
        #         d_model=dim,
        #         d_state=16,
        #         ssm_ratio=2.0,
        #         dt_rank="auto",
        #         act_layer=nn.SiLU,
        #         # ==========================
        #         d_conv=3,
        #         conv_bias=True,
        #         # ==========================
        #         dropout=0.0,
        #         # bias=False,
        #         # ==========================
        #         # dt_min=0.001,
        #         # dt_max=0.1,
        #         # dt_init="random",
        #         # dt_scale="random",
        #         # dt_init_floor=1e-4,
        #         initialize="v0",
        #         # ==========================
        #         forward_type="v2",
        #         channel_first=False,
        #     )

        # self.attn = Wave2D(res=res, dim=dim, hidden_dim=dim, infer_mode=False)

        # self.embed_len = feature_size * feature_size
        # base_window = self.embed_len // (self.sr_ratio ** 2) if self.sr_ratio > 1 else self.embed_len
        # self.window_size = int(round(math.sqrt(base_window))) ** 2
        
        # self.attn = MHLA(
        #     dim=dim,
        #     heads=num_heads,  # 对应原num_heads
        #     qkv_bias=qkv_bias,
        #     embed_len=self.embed_len,
        #     window_size=self.window_size,
        #     transform="cos",  # 可选：cos/exp/local，建议先用cos
        #     local_thres=1.5,
        #     exp_sigma=3,
        #     dropout=drop  # 对应原attn_drop
        # )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.CA_FFN = ChannelAggregationFFN(embed_dims=dim, ffn_ratio=mlp_ratio, ffn_drop=drop)

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
        B, N, C = x.shape
        # x = x + self.drop_path(self.attn(self.norm1(x).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        #                        .permute(0, 2, 3, 1).contiguous().reshape(B, N, C))


        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # B, N, C = x.shape
        # # (B, N, C) → (B, C, H, W)
        # x_2d = self.norm2(x).permute(0, 2, 1).reshape(B, C, H, W)
        # x_2d = self.CA_FFN(x_2d)
        # # (B, C, H, W) → (B, N, C)
        # x_2d = x_2d.flatten(2).transpose(1, 2)
        # x = x + self.drop_path(x_2d)

        # MHLA   [B,N,C] → [B,N_pieces,window_size,C]）

        # norm_x = self.norm1(x)  # 先做Norm1（和原逻辑一致）
        
        
        # if N % self.window_size != 0:
        #     pad_len = self.window_size - (N % self.window_size)
        #     norm_x = torch.cat([norm_x, torch.zeros(B, pad_len, C, device=norm_x.device)], dim=1)
        #     N_padded = N + pad_len
        # else:
        #     N_padded = N
        
        # N_pieces = N_padded // self.window_size  # 计算块数
        # mhla_input = norm_x.reshape(B, N_pieces, self.window_size, C)  # 重排形状
        
        # mhla_out = self.attn(mhla_input)  # 输出：[B,N_pieces,window_size,C]
        # mhla_out = mhla_out.reshape(B, N_padded, C)[:, :N, :]  # 仅保留前N个Token
        
        # x = x + self.drop_path(mhla_out)
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

# 𝑈𝑝𝑠𝑎𝑚𝑙e
class BilinearUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(BilinearUp, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear')
        self.finalconv2 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.finalrelu1 = nn.LeakyReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(in_channels // 2, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.finalconv2(x)
        x = self.finalrelu1(x)
        x = self.finalconv3(x)
        x = self.finalrelu1(x)
        return x

class EdgeFusion(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(EdgeFusion, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        y1 = self.conv1(y)

        return self.conv_out(x + x1 + y + y1)