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
                dim = embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_Classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        # Perform forward pass separately on each resolution input.
        # The inputs corresponding to a single resolution are clubbed and single
        # forward is run on the same resolution inputs. Hence we do several
        # forward passes = number of different resolutions used. We then
        # concatenate all the output features.
    
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_count=True,
            )[1],0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self,forward_features(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

        return self.head(outout) 

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cld_token, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = s + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        if self.norm in not None:
            x = self.norm(x)

        return x[:, 0]


    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embebd = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0,3,1,2),
                scale_factor = math.sqrt(npatch / N),
                mode='bicubic',
            )

        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

    def forward_selfattention(self, x):
        B, nc, w, h = x.shape
        N = self.pos_embed.shape[1] - 1
        x = self.patch_embed(x)
        
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        patch_pos_embed = nn.functional.interploate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                    dim).premute(0,3,1,2))

        if w0 != patch_pos_embed.shape[-2]:
            helper = torch.zeros(h0)[None, None, None, :].repeat(1, dim, w0 - patch_pos_embed.shape[-2], 1).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-2)

        if h0 != patch_pos_embed.shape[-1]:
            helper = torch.zeros(w0)[None, None, :, None].repeat(1, dim, 1, h0 - patch_pos_embed.shape[-1]).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-1)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    
    def forward_return_n_last_blocks(self, x, n=1, return_patch_avgpool=False):
        B = X.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_Drpo(x)

        output = []
        for i, bkl in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x)[:,0])

        if return_patch_avgpool:
            x = self.norm(x)
            output.append(torch.mean(x[:, 1:], dim=1))
        return torch.cat(output, dim=1)


