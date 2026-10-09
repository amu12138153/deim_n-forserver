import torch
import torch.nn as nn
from ..core import register


# ==========================
# Water Token Encoder
# ==========================

from .WaterQueryCrossAttention import WaterTokenEncoder as QueryWaterTokenEncoder

class WaterTokenEncoder(QueryWaterTokenEncoder):
    """Use the shared three-sensor encoder for this existing fusion stage."""
    pass


# ==========================
# Water Guided Cross Attention
# ==========================
@register()
class WaterCrossAttentionforencoder(nn.Module):
    def __init__(
        self,
        in_channels=112,
        embed_dim=64,
        num_heads=4,
        ffn_ratio=4,
        dropout=0.0,
        init_scale=0.0
    ):
        super().__init__()

        self.visual_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=1
        )

        self.water_encoder = WaterTokenEncoder(
            num_params=3,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=dropout
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        hidden_dim = int(embed_dim * ffn_ratio)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.visual_norm = nn.LayerNorm(embed_dim)
        self.water_norm = nn.LayerNorm(embed_dim)
        
        self.visual_recover = nn.Conv2d(
            embed_dim,
            in_channels,
            kernel_size=1
        )
#       self.attn_scale = nn.Parameter(
#           torch.zeros(1) + init_scale
#       )
#
#       self.ffn_scale = nn.Parameter(
#           torch.zeros(1) + init_scale
#       )
        # 新增：残差融合强度，初始化为 0
#       self.gamma = nn.Parameter(torch.zeros(1))    
        self.attn_scale = nn.Parameter(torch.ones(1) * init_scale)
        self.ffn_scale = nn.Parameter(torch.ones(1) * init_scale)
        self.gamma = nn.Parameter(torch.ones(1) * init_scale)        
        
        
    def forward(self, visual_feat, water,return_attn=False):
        """
        visual_feat:
            [B,512,40,40]

        water:
            [B,3]

        return:
            [B,512,40,40]
        """

        # 保存原始视觉特征
        identity = visual_feat
        water_tokens = self.water_encoder(water)

        # 512 -> 64
        feat = self.visual_proj(visual_feat)

        B, C, H, W = feat.shape

        # [B,64,H,W] -> [B,H*W,64]
        visual_tokens = (
            feat
            .flatten(2)
            .transpose(1, 2)
        )

        # [B,3,64]
        q = self.visual_norm(visual_tokens)
        w = self.water_norm(water_tokens)
        # cross attention
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=w,
            value=w,
            need_weights=return_attn,
            average_attn_weights=True
        )

        visual_tokens = visual_tokens + self.attn_scale * attn_out
        # FFN
        ffn_out = self.ffn(visual_tokens)

        visual_tokens = visual_tokens + self.ffn_scale * ffn_out

        # [B,H*W,64] -> [B,64,H,W]
        fused_feat = (
            visual_tokens
            .transpose(1, 2)
            .reshape(B, C, H, W)
        )

        # 64 -> 512
        delta = self.visual_recover(fused_feat)

        # 外部残差融合
        out = identity + self.gamma * delta

        if return_attn:
            return out, attn_weights

        return out


# ==========================
# Test
# ==========================
