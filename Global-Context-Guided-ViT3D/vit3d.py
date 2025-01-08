"""

3D ViT transformer that inputs 5D (n_batches, n_channels, height, weight, depth)

Based primarily on a video tutorial from Vision Transformer

and 

Official code PyTorch implementation from CDTrans paper:
https://github.com/CDTrans/CDTrans

"""

import math
import copy
from functools import partial
from itertools import repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch import Tensor, nn

import math
from typing import Tuple, Type


from utils.weight_init import trunc_normal_, init_weights_vit_timm, get_init_weights_vit, named_apply
from utils.utils import get_3d_sincos_pos_embed


# add by bryce
class MLPBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
class TwoWayAttentionBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention_for_TwoWayCrossAttn(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention_for_TwoWayCrossAttn(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention_for_TwoWayCrossAttn(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, text_embed: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            if query_pe:
                q = queries + query_pe
            else:
                q = queries
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        if query_pe and key_pe:
            q = queries + query_pe
            k = keys + key_pe
        else:
            q = queries 
            k = keys
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        
        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        if query_pe and key_pe:
            q = queries + query_pe
            k = keys + key_pe
        else:
            q = queries 
            k = keys
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

class Attention_for_TwoWayCrossAttn(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class TwoWayCrossAttn(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock3D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention_for_TwoWayCrossAttn(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        # # add by bryce
        # self.lora_k = LoRALayer(embedding_dim, embedding_dim, 64, 0.25)
        # self.lora_q = LoRALayer(embedding_dim, embedding_dim, 64, 0.25)

    def forward(
        self,
        image_embedding: Tensor,
        point_embedding: Tensor,
        text_embed: Tensor, # add by bryce
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.unsqueeze(1)
        point_embedding = point_embedding.unsqueeze(1)
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=None,
                key_pe=None,
                text_embed=text_embed
            )

        # Apply the final attention layer from the points to the image
        # q = queries + point_embedding
        # k = keys + image_pe

        # q = queries
        # k = keys

        # add by bryce
        # q = self.lora_q(q, text_embed)
        # k = self.lora_k(k, text_embed)

        # attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        # queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
### end ###

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


class PatchEmbed3D(nn.Module):
    """
    Split image into 3D patches and then embed them.

    Parameters
    ----------
    img_size : int (square)
    patch_size : int (square)
    in_chans : int
    embed_dim : int

    Atttributes:
    -----------
    n_patches : int
    proj : nn.Conv2d

    """
    def __init__(self, img_size, patch_size, embed_dim=768, patch_embed_fun='conv3d'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)

        # sample random tensor to calculate the output shape
        sample_torch = torch.rand((1, 1, *self.img_size)) # --> e.g. (1,1,128,128,128)

        if patch_embed_fun == 'conv3d':
            self.proj = nn.Conv3d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        
        out = self.proj(sample_torch)
        self.n_patches = out.flatten(2).shape[2]

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        x = self.proj(x) # out: (n_samples, embed_dim, n_patches[0], n_patches[1], n_patches[2])
        # x = x.view(-1, self.e)
        x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)///

        return x

class Attention(nn.Module):
    """
    Attention mechanism

    Parameters
    -----------
    dim : int (dim per token features)
    n_heads : int
    qkv_bias : bool
    attn_p : float (Dropout applied to q, k, v)
    proj_p : float (Dropout applied to output tensor)

    Attributes
    ----------
    scale : float
    qkv : nn.Linear
    proj : nn.Linear
    attn_drop, proj_drop : nn.Dropout
    
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, n_patches + 1, dim)

        Returns:
        -------
        Shape (n_samples, n_patches + 1, dim)

        """
        n_samples, n_tokens, dim =  x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)

        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # each with (n_samples, n_heads, n_patches + 1, head_dim)

        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Parameters
    ----------
    in_features : int
    hidden_features : int
    out_features : int
    p : float

    Attributes
    ---------
    fc1 : nn.Linear
    act : nn.GELU
    fc2 : nn.Linear
    drop : nn.Dropout
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, in_features)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(
            x
            ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x) # (n_samples, n_patches + 1, out_features)

        return x

class Block(nn.Module):
    """
    Transformer block

    Parameters
    ----------
    dim : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes
    ----------
    norm1, norm2 : LayerNorm
    attn : Attention
    mlp : MLP
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0., p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, dim)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Vision_Transformer3D(nn.Module):
    """
    3D Vision Transformer

    Parameters
    -----------
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes:
    -----------
    patch_embed : PatchEmbed
    cls_token : nn.Parameter
    pos_emb : nn.Parameter
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
    norm : nn.LayerNorm
    """
    def __init__(self, 
                img_size=384, 
                patch_size=16, 
                in_chans=3, 
                n_classes=1000, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                drop_path_rate=0.,
                p=0., 
                attn_p=0.,
                patch_embed_fun='conv3d',
                weight_init='',
                global_avg_pool=False,
                pos_embed_type='learnable',
                use_separation=True
                ):
        super().__init__()

        if patch_embed_fun in ['conv3d']:
            self.patch_embed = PatchEmbed3D(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun
            )
            
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) if global_avg_pool == False else None
        embed_len = self.patch_embed.n_patches if global_avg_pool else 1 + self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=True
            )
        
        if pos_embed_type == 'abs':
            self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=False
            )
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(np.cbrt(self.patch_embed.n_patches)), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            print('Abs pos embed built.')
            
        self.pos_drop = nn.Dropout(p=p)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=self.dpr[ii],
                    p=p,
                    attn_p=attn_p
                )
                for ii in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.pos_embed, std=.02)

        # self.apply(self._init_weights_vit_timm)

        self.init_weights(weight_init)
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
        print("Model weights initialized")

    def _init_weights_vit_timm(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def forward(self, x):
        """
        Input
        -----
        Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_classes)
        
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(
                n_samples, -1, -1
            ) # (n_samples, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        # just the CLS token
        cls_token_final = x[:, 0] if self.cls_token is not None else x.mean(dim=1)
        # cls_token_final = self.bottleneck(cls_token_final)
        x = self.head(cls_token_final)

        return x
    
    def save(self, optimizer, scaler, checkpoint):
        state = {"net": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()}
        torch.save(state, checkpoint)

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class DeiT_Transformer3D(nn.Module):
    """
    3D Vision DeiT Transformer

    Parameters
    -----------
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes:
    -----------
    patch_embed : PatchEmbed
    cls_token : nn.Parameter
    pos_emb : nn.Parameter
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
    norm : nn.LayerNorm
    """
    def __init__(self, 
                img_size=384, 
                patch_size=16, 
                n_classes=1000, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                drop_path_rate=0.,
                p=0., 
                attn_p=0.,
                patch_embed_fun='conv3d',
                weight_init='',
                training=True,
                with_dist_token=True
                ):
        super().__init__()

        if patch_embed_fun in ['conv3d', 'unet3d', 'mype3d']:
            self.patch_embed = PatchEmbed3D(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun
            )
        
        self.training = training
        self.with_dist_token = with_dist_token
        
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        if with_dist_token:
            self.dist_token = nn.Parameter(torch.rand(1, 1, embed_dim))  # new distillation token
            embed_len = self.patch_embed.n_patches + 2
            print("With distillation token")
        else:
            self.dist_token = None
            embed_len = self.patch_embed.n_patches + 1
            print("No distillation token")
        
        self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=True
            )
        
        self.pos_drop = nn.Dropout(p=p)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=self.dpr[ii],
                    p=p,
                    attn_p=attn_p
                )
                for ii in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # for w/o M3D
        # self.head = nn.Linear(embed_dim, n_classes) # changed by bryce

        # for cat with M3D
        # self.mlp = nn.Sequential(nn.Linear(embed_dim*2, 512),
        #                           nn.Dropout(0.2),
        #                           nn.ReLU(),
        #                           nn.Linear(512, 256),
        #                           nn.Dropout(0.2),
        #                           nn.ReLU(),
        #                           )
        # self.proj = nn.Linear(embed_dim, embed_dim)
        # self.norm_for_proj = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(256, n_classes) # add by bryce

        # for twoway cross attn with M3D
        self.twowayCrossAttn = TwoWayCrossAttn(3, 768, 4, 256)

        self.head = nn.Linear(embed_dim, n_classes) # add by bryce
        trunc_normal_(self.cls_token, std=.02)
        
        if with_dist_token:
            trunc_normal_(self.dist_token, std=.02)
        # trunc_normal_(self.pos_embed, std=.02)
        # self.apply(self._init_weights_vit_timm)
        
        self.init_weights(weight_init)

        # self.head_dist = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()
        # self.head_dist.apply(self._init_weights)
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.dist_token is not None:
            nn.init.normal_(self.dist_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
        print("Model weights initialized")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_vit_timm(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def forward(self, x, M3D_CLIP_visual_features):
        """
        Input
        -----
        Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_classes)
        
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        if self.with_dist_token:
            dist_token = self.dist_token.expand(n_samples, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1) # (n_samples, 2 + n_patches, embed_dim)
        else:
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        # just the CLS token
        cls_token_final = x[:, 0]
        
        # if self.with_dist_token:    
        #     dist_token_final = x[:, 1]
        #     x_dist = self.head_dist(dist_token_final)
        # else:
        #     x_dist = self.head_dist(cls_token_final)

        # for cat M3D
        # print(M3D_CLIP_visual_features.shape)
        # M3D_CLIP_visual_features = self.norm_for_proj(self.proj(M3D_CLIP_visual_features)) # (8, 768)
        # cat_features = torch.cat((cls_token_final, M3D_CLIP_visual_features), dim=-1) # (8, 768*2)
        # x = self.head(self.mlp(cat_features))
        
        # for twowayCrossAttn M3D
        if M3D_CLIP_visual_features is not None:
            fusion_feat, fusion_key = self.twowayCrossAttn(cls_token_final, M3D_CLIP_visual_features, text_embed=None)
            x = self.head(fusion_feat).squeeze(1)
        else:
        # for w/o M3D
            x = self.head(cls_token_final)
        
        if self.training:
            return x # , x_dist
        else:
            # during inference, return the average of both classifier predictions
            return x #, (x + x_dist) / 2
    
    def save(self, optimizer, scaler, checkpoint):
        state = {"net": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()}
        torch.save(state, checkpoint)