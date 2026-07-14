import torch
import torch.nn as nn
from sympy import false
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.ChangeDINO.BIE_ChangeDINO import BIE_ChangeDINO, Pre_Post_TemporalBIE
from models.EATDernet import EATDer
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.EGCTNet import EGCTNet
from models.ICIFNet import ICIFNet
from models.DMINet import DMINet, CrossAtt, BasicConv2d, decode
from models.ChangeViT import ChangeViT
from models.VcT import Reliable_Transformer
from models.RCDT import RCDT
from models.HSANet import HSANet
from models.ChangeDINO.ChangeDINO import ChangeModel
from models.ChangeMambaBCD.ChangeMambaBCD import ChangeMambaBCD
from models.CTD_Former import CTDModel
from models.EGRCNN import UNet_mtask
from models.EGPNet.network import Fuse_Unet_Edge
from models.EATDernet import EATDer
from models.BGSNet import BGSNet

# from models.lenet4 import MM_LENet4, LENetEncoder

from models.B2CNet import B2CNet
from models.AERNet.model.network import AERNet
from models.ChangeFormer import ChangeFormerV6

from models.BIE_EdgeNet import BIE_EdgeNet
from models.BIE_EdgeNet_DINO import BIE_EdgeNet_DINO

from models.re_diffatts import *


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02, pretrained_backbone_name='dino'):
    """
    迁移学习专用初始化：只初始化新增层，跳过DINO预训练主干
    pretrained_backbone_name: 预训练主干的层名前缀（根据你的模型实际命名调整）
    """
    def init_func(m):
        classname = m.__class__.__name__
        # 1. 跳过预训练主干层（根据你模型中DINO主干的层名前缀判断，比如'dino_backbone'、'vit'等）
        # 先获取当前层的完整名称（需要给模型层命名，或通过parent判断）
        # 简化方案：如果层属于DINO主干（可通过层名包含特定关键词判断），直接跳过
        layer_name = ''
        for name, module in net.named_modules():
            if module is m:
                layer_name = name
                break
        # 跳过DINO预训练主干的层（关键词根据你的模型实际调整，比如'dino'、'vit'、'backbone'）
        if pretrained_backbone_name in layer_name.lower():
            return

        # 2. 只对需要初始化的层（卷积、线性、转置卷积）进行处理
        if any(cls in classname for cls in ['Conv', 'Linear', 'ConvTranspose']):
            if hasattr(m, 'weight') and m.weight is not None:
                if init_type == 'normal':
                    init.normal_(m.weight.detach(), 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.detach(), gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.detach(), gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.detach(), 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.
#
#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
#
#     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
#     work better for some applications. Feel free to try yourself.
#     """
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, 0.0)
#
#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'EGPNet':
        define_type = 'kaiming'
    elif args.net_G == 'EATDer':
        define_type = 'xavier'
    else:
        define_type = init_type

    if args.net_G == 'BIT_ResNet':
        net = ResNet_BIT(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'BIT_ResNet1':
        net = ResNet_BIT1(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    # ========================================
    # Contrast Experiment
    # ========================================

    elif args.net_G == 'BIT':
        # base_transformer_pos_s4_dd8
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'ChangeFormer':
        # ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
        net = ChangeFormerV6(embed_dim=args.embed_dim)

    elif args.net_G == 'ChangeMamba':
        net = ChangeMambaBCD(patch_size=4,
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

    elif args.net_G == 'ChangeDINO':
        net = ChangeModel(
            backbone="mobilenetv2",
            # backbone="resnet34",
            fpn_name="fpn",
            fpn_channels=128,
            deform_groups=4,
            gamma_mode="SE",
            beta_mode="contextgatedconv",
            n_layers=[1, 1, 1, 1],
            extract_ids=[5, 11, 17, 23],
        )

    elif args.net_G == 'EGCTNet':
        net = EGCTNet(img_size=args.img_size, input_nc=3, output_nc=2, embed_dim=args.embed_dim, num_classes=args.n_class)

    elif args.net_G == 'AERNet':
        net = AERNet(pretrained=True)
    # Feature Interaction
    elif args.net_G == 'ICIFNet':
        net = ICIFNet(pretrained=True)

    elif args.net_G == 'DMINet':
        net = DMINet(pretrained=True)

    elif args.net_G == 'ChangeViT':
        net = ChangeViT(model_type='small')

    elif args.net_G == 'VcT':
        net = Reliable_Transformer(input_nc=3, output_nc=2, resnet_stages_num=4,
                                   with_pos='learned', enc_depth=1, dec_depth=8, backbone='resnet34')

    elif args.net_G == 'RCDT':
        net = RCDT(num_classes=2)

    elif args.net_G == 'HSANet':
        net = HSANet()

    elif args.net_G == 'B2CNet':
        net = B2CNet()

    elif args.net_G == 'CTDFormer':
        net = CTDModel()

    elif args.net_G == 'EGRCNN':
        net = UNet_mtask.U_Net(3, 2, 256)

    elif args.net_G == 'EGPNet':
        net = Fuse_Unet_Edge()

    elif args.net_G == 'EATDer':
        net = EATDer()

    # elif args.net_G == 'LENet':
    #     net = MM_LENet4(num_classes=2)
    elif args.net_G == 'BGSNet':
        net = BGSNet(64, num_classes=2, pretrained_path=r"F:\BIE-EdgeNet-main\models\pretrained\pvt_v2_b1.pth").cuda()


    # Ours
    elif args.net_G == 'BIE_EdgeNet':
        net = BIE_EdgeNet(img_size=args.img_size, input_nc=3, output_nc=2, embed_dim=args.embed_dim, num_classes=args.n_class)

    elif args.net_G == 'BIE_EdgeNet_DINO':
        net = BIE_EdgeNet_DINO(img_size=args.img_size, input_nc=3, output_nc=2, embed_dim=args.embed_dim, num_classes=args.n_class)

    elif args.net_G == 'BIE_ChangeDINO':
        net = BIE_ChangeDINO(
            backbone="mobilenetv2",
            # backbone="resnet34",
            fpn_name="fpn",
            fpn_channels=128,
            deform_groups=4,
            gamma_mode="SE",
            beta_mode="contextgatedconv",
            n_layers=[1, 1, 1, 1],
            extract_ids=[5, 11, 17, 23],
        )

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)

    return init_net(net, define_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################

class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)


        x = torch.abs(x1 - x2)

        if not self.if_upsample_2x:
            x = self.upsamplex2(x)

        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)

        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class ResNet_BIT(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet34',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_BIT, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

        self.test_mode = False

        # DMINet
        self.cross = CrossAtt(32, 32)
        self.fam = decode(32, 32, 32)

    # def set_test_mode(self, mode=True):
    #     """设置测试模式，控制是否返回特征"""
    #     self.test_mode = mode

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        cur1 = x1
        cur2 = x2

        # DMINet
        cross_result, cur1, cur2, attn_matrix1, attn_matrix2 = self.cross(x1, x2)
        x = self.fam(cross_result, torch.abs(cur1 - cur2))

        # x = torch.abs(cur1 - cur2)

        if not self.if_upsample_2x:
            x = self.upsamplex2(x)

        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)

        if self.test_mode:
            return x, attn_matrix1, attn_matrix1
        else:
            return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x

class ResNet_BIT1(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet34',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_BIT1, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

        self.test_mode = False

        # DMINet
        # self.cross = CrossAtt(32, 32)
        # self.fam = decode(32, 32, 32)

        # Diff-Transformer
        # self.Sem_diff = nn.Sequential(
        #     *[TransformerBlock(
        #         dim=32,
        #         spatial_attn_type="OCDA",
        #         window_size=8,
        #         overlap_ratio=0.5,
        #         num_channel_heads=8,
        #         num_spatial_heads=4,
        #         depth=1,
        #         ffn_expansion_factor=2,
        #         bias=False,
        #         LayerNorm_type="BiasFree",
        #     )
        #         for _ in range(1)]
        # )
        self.temporal_bie = Pre_Post_TemporalBIE(nf=32, heads=4, reduction=2)

    # def set_test_mode(self, mode=True):
    #     """设置测试模式，控制是否返回特征"""
    #     self.test_mode = mode

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)



        enhanced_x1, enhanced_x2, diff_feat, weight1_without, weight1, weight2_without, weight2 = self.temporal_bie(x1, x2)

        cur1 = weight1
        cur2 = weight2
        cur3 = weight1_without
        cur4 = weight2_without

        x = torch.abs(enhanced_x1 - enhanced_x2)
        x = self.upsamplex4(x)
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)

        if self.test_mode:
            return x, cur1, cur2, cur3, cur4
        else:
            return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        outputs = []
        outputs.append(x)
        return outputs

if __name__ == '__main__':
    # res = InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=1, skip_connection=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = ResNet_BIT(input_nc=3, output_nc=2, output_sigmoid=False).to(device)
    x1 = torch.randn(16, 3, 256, 256).to(device)
    x2 = torch.randn(16, 3, 256, 256).to(device)
    out = Net(x1, x2)
    print(out.shape)

