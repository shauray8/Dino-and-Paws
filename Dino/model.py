import torch 
import torch.nn as nn
import math
import numpy as np


class ViT(nn.Module):
    def __init__(self, img_size=[244], patch_size=16, channels=3, num_classes=0, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0,
            attn_drop_rate=0, drop_path_rate=0., norm_layer = nn.LayerNorm, **kwargs):

        super(ViT, self).__init__()
        self.num_features = self.embed_dim = embed_dim

        seld.patch_embed = PatchEmbed(
                img_size = img_size[0], patch_size = patch_size, channels = channels, 
                embed_dim = embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameters(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameters(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linespace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim = embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)])



