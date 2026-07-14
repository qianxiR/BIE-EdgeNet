import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from torchvision.models.resnet import ResNet34_Weights  # 确保导入ResNet34的权重类


class MMSEGBackbone(nn.Module):
    """修复通道不匹配：严格适配ResNet34的通道数"""
    def __init__(self, pretrained=True):
        super().__init__()
        # 1. 模型与权重严格匹配：ResNet34 + ResNet34_Weights
        self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # 2. 特征提取器结构（ResNet34的层结构）
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
            self.backbone.maxpool, self.backbone.layer1,
            self.backbone.layer2, self.backbone.layer3, self.backbone.layer4
        )
        # 3. ResNet34的准确通道数（layer2=128, layer3=256, layer4=512）
        self.raw_channels = [128, 256, 512]
        # 4. FPN降维：每个卷积层的输入通道严格对应raw_channels
        self.fpn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, bias=False),  # 输入128→输出256
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),  # 输入256→输出256
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),  # 输入512→输出256
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])

    def forward(self, x1, x2):
        def _extract_feat(x):
            feats = []
            x = self.feature_extractor[0:5](x)  # conv1→layer1（1/4尺度）
            for i, layer in enumerate(self.feature_extractor[5:8]):  # layer2→layer4
                x = layer(x)
                x_fpn = self.fpn[i](x)  # FPN降维到256通道
                feats.append(x_fpn)
            return feats

        feat1 = _extract_feat(x1)  # T1特征
        feat2 = _extract_feat(x2)  # T2特征
        return feat1, feat2


# -------------------------- 以下模块无需修改（保持原样） --------------------------
class ConcatFusion(nn.Module):
    def __init__(self, in_channels=[256, 256, 256]):
        super().__init__()
        self.fusion = nn.ModuleList([])
        for in_ch in in_channels:
            self.fusion.append(
                nn.Sequential(
                    nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x1, x2):
        y = []
        for idx, (feat1, feat2) in enumerate(zip(x1, x2)):
            feat = torch.cat((feat1, feat2), dim=1)
            feat = self.fusion[idx](feat)
            y.append(feat)
        return tuple(y)


class OCA(nn.Module):
    def __init__(self, in_dim=256, mid_dim=32):
        super(OCA, self).__init__()
        self.temperature = 1e-9
        self.K = nn.Linear(in_dim, mid_dim, bias=False)
        self.Q = nn.Linear(in_dim, mid_dim, bias=False)
        self.V = nn.Linear(in_dim, in_dim, bias=False)
        self.linear = nn.Linear(in_dim, in_dim, bias=False)
        self.init_weights()

    def init_weights(self):
        for m in self.K.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.Q.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.V.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None):
        x2_k = self.K(key)
        x2_k = F.normalize(x2_k, p=2, dim=-1)
        x2_k = x2_k.permute(1, 2, 0)

        x1_q = self.Q(query)
        x1_q = F.normalize(x1_q, p=2, dim=-1)
        x1_q = x1_q.permute(1, 0, 2)

        corr = F.softmax(torch.bmm(x1_q, x2_k), dim=-1)
        corr = corr / (self.temperature + corr.sum(dim=1, keepdim=True))

        x2_v = self.V(value)
        x2_v = x2_v.permute(1, 0, 2)

        y = torch.bmm(corr, x2_v).permute(1, 0, 2)
        y = self.linear(query - y)

        return F.relu(y)


class RCAM(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0, head_dim=32):
        super(RCAM, self).__init__()
        self.n = nhead
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.ModuleList([OCA(d_model, head_dim) for _ in range(self.n)])
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model * self.n, self.d_model * 4),
            nn.GELU(),
            self.dropout,
            nn.Linear(self.d_model * 4, self.d_model)
        )
        self.norm = nn.LayerNorm(self.d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query=None, key=None, value=None, pos_1=None, pos_2=None):
        query = self.with_pos_embed(query, pos_1)
        key = self.with_pos_embed(key, pos_2)
        value = self.with_pos_embed(value, pos_2)

        for i in range(self.n):
            if i == 0:
                concat = self.head[i](query, key, value)
            else:
                concat = torch.cat((concat, self.head[i](query, key, value)), dim=-1)

        y = self.ffn(concat)
        y = self.norm(y)
        return y


class RCDT(nn.Module):
    def __init__(self,
                 num_classes=2,
                 pretrained=True,
                 nhead=8,
                 dropout=0.0,
                 head_dim=32):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = 256
        self.scales = [32, 16, 8]

        self.backbone = MMSEGBackbone(pretrained=pretrained)
        self.fusion = ConcatFusion(in_channels=[self.d_model] * 3)
        self.rcam_list = nn.ModuleList([
            RCAM(d_model=self.d_model, nhead=nhead, dropout=dropout, head_dim=head_dim)
            for _ in range(3)
        ])

        self.scale_pred_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.d_model, num_classes, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_classes)
            ) for _ in range(3)
        ])

        self.final_fusion = nn.Sequential(
            nn.Conv2d(self.d_model * 3, self.d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(self.d_model, self.d_model // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model // 2, num_classes, kernel_size=1)
        )

    def _get_pos_embed(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000) / d_model))
        pos_embed = torch.zeros(1, seq_len, d_model, device=device)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        assert H == 256 and W == 256, "论文默认输入256×256"

        feat1, feat2 = self.backbone(x1, x2)
        fused_feats = self.fusion(feat1, feat2)

        rcam_feats = []
        multi_scale_preds = []
        for i, (scale, fused_feat) in enumerate(zip(self.scales, fused_feats)):
            S = scale
            seq_len = S * S

            t1_seq = feat1[i].view(B, self.d_model, seq_len).transpose(1, 2)
            t2_seq = feat2[i].view(B, self.d_model, seq_len).transpose(1, 2)

            pos_embed = self._get_pos_embed(seq_len, self.d_model, t1_seq.device)

            rcam_out = self.rcam_list[i](
                query=t1_seq, key=t2_seq, value=t2_seq, pos_1=pos_embed, pos_2=pos_embed
            )

            rcam_feat = rcam_out.transpose(1, 2).reshape(B, self.d_model, S, S)
            rcam_feats.append(rcam_feat)

            scale_pred = self.scale_pred_heads[i](rcam_feat)
            multi_scale_preds.append(scale_pred)

        rcam_32 = rcam_feats[0]
        rcam_16_up = F.interpolate(rcam_feats[1], size=32, mode='bilinear', align_corners=False)
        rcam_8_up = F.interpolate(rcam_feats[2], size=32, mode='bilinear', align_corners=False)
        concat_feat = torch.cat([rcam_32, rcam_16_up, rcam_8_up], dim=1)
        final_pred = self.final_fusion(concat_feat)

        return multi_scale_preds, final_pred