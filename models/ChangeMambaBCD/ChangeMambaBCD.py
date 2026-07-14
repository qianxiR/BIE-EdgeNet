import sys
import os
sys.path.append("C:\\Users\\Administrator\\VMamba\\kernels\\selective_scan\\build\\lib.win-amd64-3.10")

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（BIE-EdgeNet-main）
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
# 将根目录加入sys.path
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .Mamba_backbone import Backbone_VSSM
# from .vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
# from .ChangeDecoder import ChangeDecoder

from models.ChangeMambaBCD.Mamba_backbone import Backbone_VSSM
from models.ChangeMambaBCD.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from models.ChangeMambaBCD.ChangeDecoder import ChangeDecoder
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count



class ChangeMambaBCD(nn.Module):
    def __init__(self, pretrained=None,
                 patch_size=4,
                 in_chans=3,
                 num_classes=2,
                 depths=[2, 2, 9, 2],
                 dims=96,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer="silu",
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.0,
                 ssm_init="v0",
                 forward_type="v2",
                 # MLP 相关参数（配置默认值）
                 mlp_ratio=4.0,
                 mlp_act_layer="gelu",
                 mlp_drop_rate=0.0,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 norm_layer="ln",
                 downsample_version="v2",
                 patchembed_version="v2",
                 gmlp=False,
                 use_checkpoint=False):
        super(ChangeMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            norm_layer=norm_layer,
            downsample_version=downsample_version,
            patchembed_version=patchembed_version,
            gmlp=gmlp,
            use_checkpoint=use_checkpoint
        )

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        # norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        # ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        # mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        norm_layer = _NORMLAYERS.get(norm_layer.lower(), nn.LayerNorm)
        ssm_act_layer = _ACTLAYERS.get(ssm_act_layer.lower(), nn.SiLU)
        mlp_act_layer = _ACTLAYERS.get(mlp_act_layer.lower(), nn.GELU)

        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features, post_features)

        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output

if __name__ == '__main__':
    model = ChangeMambaBCD(pretrained=None,
                           patch_size=4,
                           in_chans=3,
                           num_classes=2,
                           depths=[2, 2, 9, 2],
                           dims=96,
                           # ===================
                           ssm_d_state=16,
                           ssm_ratio=2.0,
                           ssm_rank_ratio=2.0,
                           ssm_dt_rank="auto",
                           ssm_act_layer="silu",
                           ssm_conv=3,
                           ssm_conv_bias=True,
                           ssm_drop_rate=0.0,
                           ssm_init="v0",
                           forward_type="v2",
                           # ===================
                           mlp_ratio=4.0,
                           mlp_act_layer="gelu",
                           mlp_drop_rate=0.0,
                           # ===================
                           drop_path_rate=0.1,
                           patch_norm=True,
                           norm_layer="ln",
                           downsample_version="v2",
                           patchembed_version="v2",
                           gmlp=False,
                           use_checkpoint=False
                           )

    # 自动用 GPU（有GPU就用，没有就用CPU）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"模型运行在：{device}")

    # 测试输入（和你之前的一致：1 batch, 3通道, 256x256）
    x1 = torch.randn(1, 3, 256, 256).to(device)
    x2 = torch.randn(1, 3, 256, 256).to(device)

    # 推理（关闭梯度计算，节省内存）
    with torch.no_grad():
        output1 = model(x1, x2)

    # 打印输出形状，验证模型正常运行
    print(f"模型输出形状：{output1.shape}")  # 预期：torch.Size([1, 2, 256, 256])
    print("模型运行成功！")