from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from einops import rearrange

from models.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from models.ChangeViT_resnet import resnet18, resnet34



def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            ffn_bias=True,
            proj_bias=True,
            drop_path_rate=0.0,
            drop_path_uniform=False,
            init_values=None,  # for layerscale: None or 0 => no layerscale
            embed_layer=PatchEmbed,
            act_layer=nn.GELU,
            block_fn=Block,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            print("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            print("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            print("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i: i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return x_norm

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,  # Layers or n last layers to take
            reshape: bool = False,
            return_class_token: bool = False,
            norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Encoder(nn.Module):
    def __init__(self, model_type='small'):
        super().__init__()
        if model_type == 'tiny':
            self.vit = DinoVisionTransformer(
                img_size=256,
                patch_size=16,
                embed_dim=192,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                block_fn=partial(Block, attn_class=MemEffAttention),
                num_register_tokens=0
            )
            path = "checkpoint/deit_tiny_patch16_224-a1311bcf.pth"

        elif model_type == 'small':
            self.vit = DinoVisionTransformer(
                img_size=256,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                block_fn=partial(Block, attn_class=MemEffAttention),
                num_register_tokens=0
            )
            # path = "checkpoint/dinov2_vits14_pretrain.pth"

        else:
            assert False, r'Encoder: check the vit model type'

        # state_dict = torch.load(path, map_location='cpu')['model'] \
        #     if model_type == 'tiny' else torch.load(path, map_location='cpu')
        #
        # for k in ['pos_embed', 'patch_embed.proj.weight']:
        #     del state_dict[k]
        # msg = self.vit.load_state_dict(state_dict, strict=False)
        # print(' missing_keys:{},\n unexpected_keys:{}'.format(msg.missing_keys, msg.unexpected_keys))
        # print('model_type: {},\n checkpoint_path: {}'.format(model_type, path))

        self.resnet = resnet34(pretrained=True)
        self.drop = nn.Dropout(p=0.01)

    def detail_capture(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x2 = self.drop(self.resnet.layer1(x))
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)

        return [x2, x3, x4]

    def forward(self, x, y):
        v_x = self.vit(x)
        v_y = self.vit(y)

        v_x = rearrange(v_x, 'b (h w) c -> b c h w', h=16, w=16)
        v_y = rearrange(v_y, 'b (h w) c -> b c h w', h=16, w=16)

        c_x = self.detail_capture(x)
        c_y = self.detail_capture(y)

        return c_x + [v_x], c_y + [v_y]

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class CAMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape

        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B2, N2, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CABlock(nn.Module):
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.attn = CrossAttention(dim1, dim2, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = CAMlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class FeatureInjector(nn.Module):
    def __init__(self, dim1=384, dim2=[64, 128, 256], num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.c2_c5 = CABlock(dim1, dim2[0], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c3_c5 = CABlock(dim1, dim2[1], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c4_c5 = CABlock(dim1, dim2[2], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)

        self.fuse = nn.Conv2d(dim1*3, dim1, 1, bias=False)

        weight_init(self)


    def base_forward(self, c2, c3, c4, c5):
        H, W = c5.shape[2:]

        c2 = rearrange(c2, 'b c h w -> b (h w) c')
        c3 = rearrange(c3, 'b c h w -> b (h w) c')
        c4 = rearrange(c4, 'b c h w -> b (h w) c')
        c5 = rearrange(c5, 'b c h w -> b (h w) c')

        _c2 = self.c2_c5(c5, c2)
        _c2 = rearrange(_c2, 'b (h w) c -> b c h w', h=H, w=W)

        _c3 = self.c3_c5(c5, c3)
        _c3 = rearrange(_c3, 'b (h w) c -> b c h w', h=H, w=W)

        _c4 = self.c4_c5(c5, c4)
        _c4 = rearrange(_c4, 'b (h w) c -> b c h w', h=H, w=W)

        _c5 = self.fuse(torch.cat([_c2, _c3, _c4], dim=1))

        return _c5

    def forward(self, fx, fy):
        _c5x = self.base_forward(fx[0], fx[1], fx[2], fx[3])
        _c5y = self.base_forward(fy[0], fy[1], fy[2], fy[3])

        return _c5x, _c5y


class Decoder(nn.Module):
    def __init__(self, in_dim=[64, 128, 256, 384], decay=4, num_class=1):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.up_c5 = nn.Sequential(
            nn.Conv2d(c5_channel, c4_channel, 1, bias=False),
            nn.ConvTranspose2d(c4_channel, c4_channel, kernel_size=4, stride=2, padding=1)
        )

        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, c3_channel, 1, bias=False),
            nn.ConvTranspose2d(c3_channel, c3_channel, kernel_size=4, stride=2, padding=1)
        )

        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, c2_channel, 1, bias=False),
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1)
        )

        self.classfier = nn.Sequential(
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(c2_channel, num_class, 3, 1, padding=1, bias=False)
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim*3, dim//decay, 1, bias=False),
                nn.BatchNorm2d(dim//decay),
                nn.ReLU(),
                nn.Conv2d(dim//decay, dim//decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim//decay, dim//decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim//decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

    def difference_modeling(self, x, y, block):
        f = torch.cat([x, y, torch.abs(x-y)], dim=1)
        f = block(f)

        return f

    def forward(self, fx, fy):
        c2x, c3x, c4x = fx[:-1]
        c2y, c3y, c4y = fy[:-1]

        c5x, c5y = self.structure_enhance(fx, fy)

        c2 = self.difference_modeling(c2x, c2y, self.mlp[0])
        c3 = self.difference_modeling(c3x, c3y, self.mlp[1])
        c4 = self.difference_modeling(c4x, c4y, self.mlp[2])
        c5 = self.difference_modeling(c5x, c5y, self.mlp[3])

        c4f = c4 + self.up_c5(c5)
        c3f = c3 + self.up_c4(c4f)
        c2f = c2 + self.up_c3(c3f)

        pred = self.classfier(c2f)
        pred_mask = torch.sigmoid(pred)

        return pred_mask

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            for f, g in m.named_children():
                print('initialize: ' + f)
                if isinstance(g, nn.Conv2d):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(g.weight)
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, nn.Linear):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AdaptiveMaxPool2d) or isinstance(m, nn.ModuleList) or isinstance(m, nn.BCELoss):
            a=1
        else:
            pass

class ChangeViT(nn.Module):
    def __init__(self, model_type='small'):
        super().__init__()

        self.encoder = Encoder(model_type)

        self.decoder = Decoder(in_dim=[64, 128, 256, 384])
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        pred = self.decoder(fx, fy)

        return pred

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = ChangeViT().to(device)
    x1 = torch.randn(16, 3, 256, 256).to(device)
    x2 = torch.randn(16, 3, 256, 256).to(device)
    out = Net(x1, x2)
    print([o.shape for o in out])