import torch
import torch.nn as nn
import torch.nn.functional as F
#代码地址：https://github.com/saadwazir/MCADS-Decoder/blob/main/mcadsDecoder.py
#被改进的模块 看Ai缝合怪b站视频：2025.8.10 更新的视频
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


# DSUB: Depth-to-Space Upsampling Block
class DSUB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale

        # pre-up: Conv3x3 + ReLU，把通道扩到 out_ch * r^2 以便 PixelShuffle
        self.pre_conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), kernel_size=3, padding=1, bias=True)
        self.pre_act = nn.ReLU(inplace=True)

        # D2S（PixelShuffle）
        self.ps = nn.PixelShuffle(scale)

        # post-up: Conv3x3 + ReLU（轻精炼）
        self.post_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2,padding=1, bias=True)
        self.post_act = nn.ReLU(inplace=True)

        # 最终用 CB 进一步精炼（与论文 Fig.1(d) 收尾一致）
        self.refine = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_act(x)
        x = self.ps(x)                 # H,W 放大 scale 倍，通道缩到 out_ch
        x = self.post_conv(x)
        x = self.post_act(x)
        x = self.refine(x)
        return x
# ---------------------------
# EUB: Effective Upsampling Block
# ---------------------------
class EUB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        # 第一段 CB：in_ch → out_ch
        self.cb1 = ConvBlock(in_ch, out_ch)
        # 上采样（双线性）
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        # 第二段 CB：out_ch → out_ch
        self.cb2 = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.cb1(x)
        x = self.upsample(x)
        x = self.cb2(x)
        return x

# Residual Block (RB)
class ResidualRB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.act = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = self.conv3(x)
        y = self.conv1(y)
        y = x + y
        y = self.bn(y)
        y = self.act(y)
        return y


class RLAB(nn.Module): #原模块
    """
    Residual Linear Attention Block
    - n_rb: RB 迭代次数，建议各stage从深到浅用 [5,4,3,2,1]
    - proj_dim: 注意力中的 dk/dv（不设即用通道数）
    """
    def __init__(self, in_ch_skip: int, in_ch_dec: int,
                 n_rb: int = 3, proj_dim: int | None = None):
        super().__init__()
        self.n_rb = n_rb
        # 1) 对 skip 分支做多次 RB 精炼（式(8)）
        self.rb_blocks = nn.ModuleList([ResidualRB(in_ch_skip) for _ in range(n_rb)])
        # 2) 先用 CB 精炼（与论文“Following the concatenation, the combined feature map is refined”一致）
        out_ch = in_ch_skip
        self.refine = ConvBlock(2*in_ch_skip, out_ch)

        # 3) 线性注意力：1x1投影成Q/K/V；把(H,W)展平为N
        d_model = out_ch if proj_dim is None else proj_dim
        self.q = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False)
        self.k = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False)
        self.v = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False)
        self.out = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=False)  # 回到out_ch

        self.scale = d_model ** 0.5  # sqrt(dk)

        self.eub = EUB(in_ch_skip,in_ch_skip)
        self.dsub  =DSUB(in_ch_dec,in_ch_skip)

    def forward(self, skip_feat, dec_feat):
        # RB refinement on skip
        x_rb = skip_feat
        for rb in self.rb_blocks:
            x_rb = rb(x_rb)  # 多次迭代，深层更多

        dec_feat = self.dsub(dec_feat) # 保证skip_feat和dec_feat 特征 通道数相同

        if dec_feat.shape[2] != skip_feat.shape[2]:  # 保证skip_feat和dec_feat 特征 尺寸大小相同
            dec_feat = self.eub(dec_feat)

        # 融合
        x_1 = torch.cat([x_rb, dec_feat], dim=1)  # (B, 2*C_skip, H, W)


        x_ref = self.refine(x_1)                  # (B, out_ch, H, W)

        # 线性注意力：把空间展平成 token 维
        B, C, H, W = x_ref.shape
        N = H * W

        q = self.q(x_ref).flatten(2).transpose(1, 2)  # (B, N, d)
        k = self.k(x_ref).flatten(2).transpose(1, 2)  # (B, N, d)
        v = self.v(x_ref).flatten(2).transpose(1, 2)  # (B, N, d)

        attn = torch.matmul(q, k.transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)                # (B, N, N)  —— 式(10)

        y = torch.matmul(attn, v)                     # (B, N, d)
        y = y.transpose(1, 2).reshape(B, -1, H, W)    # (B, d, H, W)
        y = self.out(y)                               # (B, out_ch, H, W)

        out = y + self._proj_if_needed(x_1, y.shape[1])  # 更稳的做法：显式投影
        return out

    # 为了稳妥，给残差分支加一个自动投影（如果通道不匹配）
    def _proj_if_needed(self, x, out_ch):
        if x.shape[1] == out_ch:
            return x
        if not hasattr(self, 'proj_res'):
            self.proj_res = nn.Conv2d(x.shape[1], out_ch, 1, bias=False)
        return self.proj_res(x)

#------------
from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)
def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)
class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        '''#来自Ai缝合怪整理
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1)  # (h w d1) #来自Ai缝合怪整理
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1)  # (h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos

class MALAttention(nn.Module): #来自Ai缝合怪整理

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor,sin:torch.Tensor, cos: torch.Tensor):
        ''' #来自Ai缝合怪整理
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape

        qkvo = self.qkvo(x)  # (b 4*c h w)
        qkv = qkvo[:, :3 * self.dim, :, :]
        o = qkvo[:, 3 * self.dim:, :, :] # b c h w

        lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])  # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)  # (b n (h w) d)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H * W)) ** 0.5) @ (v * (self.scale / (H * W)) ** 0.5)

        res = q @ kv * (1 + 1 / (z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)
class MALFM(nn.Module): #多尺度感知线性融合模块，也可以称：多尺度幅度感知线性融合模块
    """
    Residual Linear Attention Block
    - n_rb: RB 迭代次数，建议各stage从深到浅用 [5,4,3,2,1]
    - proj_dim: 注意力中的 dk/dv（不设即用通道数）
    """
    def __init__(self, in_ch_skip: int, in_ch_dec: int,
                 n_rb: int = 3, proj_dim: int | None = None):
        super().__init__()
        self.n_rb = n_rb
        # 1) 对 skip 分支做多次 RB 精炼（式(8)）
        self.rb_blocks = nn.ModuleList([ResidualRB(in_ch_skip) for _ in range(n_rb)])
        # 2) 先用 CB 精炼（与论文“Following the concatenation, the combined feature map is refined”一致）
        out_ch = in_ch_skip
        self.refine = ConvBlock(2*in_ch_skip, out_ch)

        # 3) MALA幅度感知线性注意力：来自ICCV2025 平替普通线性注意力，好好讲好故事，可以冲SCI二区及CCF-B。
        self.mala = MALAttention(out_ch)

        self.eub = EUB(in_ch_skip,in_ch_skip)
        self.dsub  =DSUB(in_ch_dec,in_ch_skip)

    def forward(self, skip_feat, dec_feat):
        # RB refinement on skip
        x_rb = skip_feat
        for rb in self.rb_blocks:
            x_rb = rb(x_rb)  # 多次迭代，深层更多

        dec_feat = self.dsub(dec_feat) # 保证skip_feat和dec_feat 特征 通道数相同

        if dec_feat.shape[2] != skip_feat.shape[2]:  # 保证skip_feat和dec_feat 特征 尺寸大小相同
            dec_feat = self.eub(dec_feat)

        # 融合
        x_1 = torch.cat([x_rb, dec_feat], dim=1)  # (B, 2*C_skip, H, W)


        x_ref = self.refine(x_1)                  # (B, out_ch, H, W)

        y = self.mala(x_ref) #在此处使用幅度感知线性注意力模块 ，ICCV2025顶会论文。

        out = y + self._proj_if_needed(x_1, y.shape[1])  # 更稳的做法：显式投影
        return out

    # 为了稳妥，给残差分支加一个自动投影（如果通道不匹配）
    def _proj_if_needed(self, x, out_ch):
        if x.shape[1] == out_ch:
            return x
        if not hasattr(self, 'proj_res'):
            self.proj_res = nn.Conv2d(x.shape[1], out_ch, 1, bias=False)
        return self.proj_res(x)

# 输入 B H W C , 输出 B  H W C
if __name__ == "__main__":
    print("第一种情况：尺寸和通道数都不相同进行特征融合")
    input_dec = torch.randn(1, 64, 32, 32)
    input_skip = torch.randn(1, 32, 64, 64)

    MALFM1 = MALFM(in_ch_skip=32,in_ch_dec=64,n_rb=3) # 创建  RLAB 模块实例
    output = MALFM1 (input_skip,input_dec)
    print("MALFM1_输入input_dec张量形状:", input_dec.shape)
    print("MALFM1_输入input_skip张量形状:", input_skip.shape)
    print("MALFM1_输出张量形状:", output.shape)

    print("\n第二种情况：尺寸和通道数都相同进行特征融合")
    input_dec2 = torch.randn(1, 64, 64, 64)
    input_skip2 = torch.randn(1, 64, 64, 64)
    MALFM2 = MALFM(in_ch_skip=64,in_ch_dec=64,n_rb=3) # 创建  RLAB 模块实例
    output = MALFM2(input_skip2,input_dec2)
    print("MALFM2_输入input_dec2张量形状:", input_dec2.shape)
    print("MALFM2_输入input_skip2张量形状:", input_skip2.shape)
    print("MALFM2_输出张量形状:", output.shape)

    print('MALFM是CVPR2025 RLAB的二次创新模块')
