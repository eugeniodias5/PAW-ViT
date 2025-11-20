""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Hacked together by / Copyright 2020 Ross Wightman
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo

import math

from functools import partial

from models.senet import seresnet50


WEIGHTS_PATH = "models/weights"

__model_urls = {
    "vit_small_patch16_224": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
    "vit_base_patch16_224": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
    "vit_base_patch16_384": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth",
    "vit_large_patch16_224": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth",
}


def load_pretrained_url(model):
    url = __model_urls[model]
    model_basename = url.split("/")[-1]

    if os.path.exists(os.path.join(WEIGHTS_PATH, model_basename)):
        pretrained_path = os.path.join(WEIGHTS_PATH, model_basename)
        return pretrained_path

    return url


def vit_small_patch16_224(
    use_bnneck,
    patch_size=16,
    stride_size=16,
    num_classes=1000,
    num_directions=2,
    img_size=(224, 224),
    pretrained=True,
    *args,
    **kwargs,
):
    if pretrained:
        pretrained = load_pretrained_url("vit_small_patch16_224")

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        stride_size=stride_size,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qkv_bias=True,
        drop_path=0.1,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        num_directions=num_directions,
        pretrained=pretrained,
        use_bnneck=use_bnneck,
    )

    return model


def vit_base_patch16_224(
    use_bnneck,
    patch_size=16,
    stride_size=16,
    num_classes=1000,
    num_directions=2,
    img_size=(224, 224),
    pretrained=True,
    *args,
    **kwargs,
):
    if pretrained:
        pretrained = load_pretrained_url("vit_base_patch16_224")

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        num_directions=num_directions,
        pretrained=pretrained,
        use_bnneck=use_bnneck,
    )

    return model


def vit_base_patch16_384(
    use_bnneck,
    patch_size=16,
    stride_size=16,
    num_classes=1000,
    num_directions=2,
    img_size=(224, 224),
    pretrained=True,
    *args,
    **kwargs,
):
    if pretrained:
        pretrained = load_pretrained_url("vit_base_patch16_384")

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        num_directions=num_directions,
        pretrained=pretrained,
        use_bnneck=use_bnneck,
    )

    return model


def vit_base_seresnet50_patch16_224(
    use_bnneck,
    patch_size=16,
    stride_size=16,
    num_classes=1000,
    num_directions=2,
    img_size=(224, 224),
    pretrained=True,
    *args,
    **kwargs,
):
    if pretrained:
        pretrained = load_pretrained_url("vit_base_patch16_224")

    # Loading SEResNet50 as patch embedding backbone
    backbone = seresnet50(layers=[3, 4, 9], pretrained=True)
    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        num_directions=num_directions,
        pretrained=pretrained,
        use_bnneck=use_bnneck,
        hybrid_backbone=backbone,
    )

    return model


def vit_large_patch16_224(
    use_bnneck,
    patch_size=16,
    stride_size=16,
    num_classes=1000,
    img_size=(224, 224),
    pretrained=True,
    num_directions=2,
    *args,
    **kwargs,
):
    if pretrained:
        pretrained = load_pretrained_url("vit_large_patch16_224")

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        stride_size=stride_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        num_directions=num_directions,
        pretrained=pretrained,
        use_bnneck=use_bnneck,
    )

    return model


def vit_base_seresnet50_patch16_224(
    use_bnneck,
    patch_size=16,
    stride_size=16,
    num_classes=1000,
    pretrained=True,
    *args,
    **kwargs,
):
    if pretrained:
        pretrained = load_pretrained_url("vit_base_patch16_224")

    # Loading SEResNet50 as patch embedding backbone
    backbone = seresnet50(layers=[3, 4, 6, 3], pretrained=True)
    # We drop layer4
    del backbone.layer4
    del backbone.last_linear

    model = ViT(
        img_size=(224, 224),
        patch_size=patch_size,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        pretrained=pretrained,
        use_bnneck=use_bnneck,
        hybrid_backbone=backbone,
    )

    return model


def vit_large_patch16_224(
    num_parts,
    use_bnneck,
    patch_size=16,
    stride_size=16,
    num_classes=1000,
    pretrained=True,
    *args,
    **kwargs,
):
    if pretrained:
        pretrained = load_pretrained_url("vit_large_patch16_224")

    model = ViT(
        img_size=(224, 224),
        patch_size=patch_size,
        stride_size=stride_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        pretrained=pretrained,
        num_parts=num_parts,
        use_bnneck=use_bnneck,
    )

    return model


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    Image to patch Embedding
    """

    def __init__(
        self, img_size=224, patch_size=16, stride_size=16, in_channels=3, embed_dim=768
    ):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.stride_size = (stride_size, stride_size)

        self.num_x = (self.img_size[1] - self.patch_size[1]) // self.stride_size[1] + 1
        self.num_y = (self.img_size[0] - self.patch_size[0]) // self.stride_size[0] + 1

        self.num_patches = self.num_x * self.num_y
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Patch embedding layer
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=stride_size
        )

        # Initializing weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # Flatten the image into patches
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding"""

    def __init__(
        self, backbone, img_size=224, backbone_size=None, in_channels=3, embed_dim=768
    ):
        super(HybridEmbed, self).__init__()
        assert isinstance(backbone, nn.Module)
        self.backbone = backbone
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Getting the size of the output feature map
        with torch.no_grad():
            x = torch.zeros(1, in_channels, img_size[0], img_size[1])
            self.backbone.eval()
            x = self.backbone.layer0(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)

            backbone_shape = x.shape
            feature_size = backbone_shape[-2:]
            feature_dim = backbone_shape[1]
            backbone.train()

        self.num_x = feature_size[0]
        self.num_y = feature_size[1]
        self.num_patches = feature_size[0] * feature_size[1]

        if feature_dim != embed_dim:
            self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ k.transpose(-2, -1)) * self.scale

        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attention


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """ "
    Vision Transformer (ViT) for Pose ReID

    This model is implemented to Animal ReID. The strategy is to use the ViT backbone to do ReID and Pose estimation,
    Enhancing performances by using the pose information to focus on the important parts of the animal.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        in_channel=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained=False,
        use_bnneck=False,
        hybrid_backbone=None,
        num_directions=2
    ):
        super(ViT, self).__init__()
        self.num_classes = num_classes
        self.num_directions = num_directions
        self.hybrid = False

        if not hybrid_backbone:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                stride_size=stride_size,
                in_channels=in_channel,
                embed_dim=embed_dim,
            )

        else:
            self.patch_embed = HybridEmbed(
                backbone=hybrid_backbone,
                img_size=img_size,
                in_channels=in_channel,
                embed_dim=embed_dim,
            )
            self.hybrid = True

        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Aggregation token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # Create a fc for the aggregation token
        self.fc = (
            nn.Linear(embed_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )
        self.fc_direction = nn.Linear(embed_dim, self.num_directions)

        self.bnneck = None
        if use_bnneck:
            self.bnneck = nn.BatchNorm1d(embed_dim)
            self.bnneck.bias.requires_grad_(False)

        self.apply(self._init_weights)

        if pretrained and isinstance(pretrained, str):
            self.load_state_dict_imagenet(pretrained)


    def prepare_tokens(self, x):
        B, N, W, H = x.shape

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        return x

    def forward(self, x, visibility=None):
        B, _, _, _ = x.shape

        x = self.prepare_tokens(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        cls = x[:, 0].clone()

        # CLS head
        if self.bnneck:
            cls = self.bnneck(cls)

        logits = self.fc(cls)
        direction = self.fc_direction(cls)

        return x[:, 0], logits.squeeze(), direction.squeeze()


    def infer(self, x):
        B, _, _, _ = x.shape

        x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]


    def load_state_dict_imagenet(self, pretrained_path):
        # Check if pretrained is an url
        print(pretrained_path)
        if pretrained_path.startswith("http") or pretrained_path.startswith("https"):
            print("Loading pretrained model from url")
            param_dict = model_zoo.load_url(pretrained_path)
        else:
            param_dict = torch.load(pretrained_path, map_location="cpu")

        if "model" in param_dict:
            param_dict = param_dict["model"]

        for i in param_dict:
            if "head" in i or "dist" in i:
                continue
            if i == "pos_embed":
                param = param_dict[i]

                # Resize the pos_embed if it doesn't match the input size
                if param.shape[1] != self.pos_embed.shape[1]:
                    param = self.resize_pos_embed(
                        param,
                        self.pos_embed,
                        self.patch_embed.num_y,
                        self.patch_embed.num_x,
                    )

                param_dict[i] = param

            if i.startswith("patch_embed") and self.hybrid:
                continue

            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])

        print("Pretrained model loaded from %s" % pretrained_path)


    def resize_pos_embed(self, posemb, posemb_new, height, width):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

        ntok_new = posemb_new.shape[1]
        
        posemb_token = posemb[:, :1]
        posemb_grid = posemb[0, 1:]

        ntok_new -= 1

        gs_old = int(math.sqrt(len(posemb_grid)))
        print(
            "Resized position embedding from size:{} to size: {} with height:{} width: {}".format(
                posemb.shape, posemb_new.shape, height, width
            )
        )
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(
            posemb_grid, size=(height, width), mode="bicubic", align_corners=True
        )
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)

        posemb = torch.cat([posemb_token, posemb_grid], dim=1)
        return posemb


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
