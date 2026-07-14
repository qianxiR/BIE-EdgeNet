import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
import yaml
from easydict import EasyDict
import os, sys

# ===================== 关键：Tree_SSM的导入路径，根据你的项目目录修改 =====================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from tree_scanning import Tree_SSM

# ===================== 你提供的所有工具类+基础层 完整复制 =====================
class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def build_norm_layer(dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last": layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last": layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first": layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first": layers.append(to_channels_first())
    else: raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == "ReLU": return nn.ReLU(inplace=True)
    elif act_layer == "SiLU": return nn.SiLU(inplace=True)
    elif act_layer == "GELU": return nn.GELU()
    raise NotImplementedError(f"build_act_layer does not support {act_layer}")

class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer="GELU", drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x);x = self.act(x);x = self.drop(x);x = self.fc2(x);x = self.drop(x)
        return x

class GraphSSMLayer(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0, drop=0.0, norm_layer="LN", drop_path=0.0, act_layer="GELU", post_norm=False, layer_scale=None, with_cp=False):
        super().__init__()
        self.channels = channels
        self.with_cp = with_cp
        self.norm1 = build_norm_layer(channels, "LN")
        self.post_norm = post_norm
        # 核心的Tree_SSM全局感知模块
        self.TreeSSM = Tree_SSM(
            d_model=channels, d_state=1, ssm_ratio=2, ssm_rank_ratio=2, dt_rank="auto", act_layer=nn.SiLU,
            d_conv=3, conv_bias=False, dropout=0.0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(channels, "LN")
        self.mlp = MLPLayer(in_features=channels, hidden_features=int(channels * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)

    def forward(self, x):
        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.TreeSSM(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.TreeSSM(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                if self.post_norm:
                    x = x + self.drop_path(self.gamma1 * self.norm1(self.TreeSSM(x)))
                    x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.gamma1 * self.TreeSSM(self.norm1(x)))
                    x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x
        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x