import torch
import torch.nn as nn
from einops import rearrange
#论文：Fast and Accurate Image Restoration and Generation with Rank Enhanced Linear Attention
#看Ai缝合怪b站视频：2026.10.3 更新的视频
class AERLA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, kernels=(3, 5, 7)):
        super().__init__()
        assert dim % num_heads == 0
        self.heads, self.eps = num_heads, 1e-6
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.dwc = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=bias)
            for k in kernels
        ])
        self.router = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, len(kernels) + 1)
        )
        self.act = nn.ELU()

    def rank_stats(self, x):  # B,H,N,D
        x = x - x.mean(2, keepdim=True)
        cov = x.transpose(-2, -1) @ x / max(x.shape[2] - 1, 1)
        d = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(self.eps)

        total = cov.pow(2).sum((-2, -1))
        redundancy = (total - d.pow(2).sum(-1)) / (total + self.eps)

        p = d / (d.sum(-1, keepdim=True) + self.eps)
        entropy = -(p * torch.log(p + self.eps)).sum(-1)
        concentration = 1 - entropy / torch.log(
            torch.tensor(float(x.shape[-1]), device=x.device)
        )

        return torch.stack([redundancy, concentration], -1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).chunk(3, 1)
        q, k = self.act(q) + 1, self.act(k) + 1

        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.heads)
        vh = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.heads)

        z = 1 / (q @ k.mean(2, keepdim=True).transpose(-2, -1)).clamp_min(self.eps)
        kv = (k.transpose(-2, -1) * N**-0.5) @ (vh * N**-0.5)
        g = q @ kv * z                                  # global feature

        r = self.router(self.rank_stats(g).detach())
        weights = torch.softmax(r[..., :-1], -1)
        gate = torch.sigmoid(r[..., -1:])

        local = 0
        for i, conv in enumerate(self.dwc):
            f = rearrange(conv(v), 'b (h d) x y -> b h d x y', h=self.heads)
            local = local + f * weights[..., i, None, None, None]

        g = rearrange(g, 'b h (x y) d -> b h d x y', x=H, y=W)
        out = g + local * gate[..., None, None]

        out = rearrange(out, 'b h d x y -> b (h d) x y')
        out = self.proj(out)

        return out

#AERLA: Adaptive Effective-Rank Linear Attention 自适应有效秩线性注意力
#可以直接拿去发论文，作为CCF-A/B/C，或是SCI一区/二区/三区/四区 的一种即插即用涨点创新模块
'''
AERLA涉及两个核心改进：
Effective Rank 表示不再只关注数学Rank，而是关注真正有效的特征多样性；
Adaptive 表示通过Rank-aware Router动态选择局部补偿尺度和强度。
'''
'''
简单描述一下：（供群里小伙伴参考！）
这个二次创新模块的动机是：原始RELA虽然通过深度卷积提升了线性注意力特征的数学Rank，
但“满秩”并不一定代表特征真正具有足够的信息多样性，特征能量仍可能集中在少数几个方向；
同时，原RELA对所有输入都采用固定大小的深度卷积进行统一补偿，缺少对不同图像、不同退化区域和不同特征状态的自适应能力。

因此，改进模块引入Effective Rank感知机制，先判断当前线性注意力输出是否存在较强的特征冗余和谱能量集中，
再通过Rank-aware Router动态选择不同尺度的局部卷积分支，并自适应控制局部补偿强度。
这样可以做到“低秩严重时加强补偿、特征已经充分时减少干预”，避免固定卷积带来的冗余增强。

其主要作用：提升线性注意力输出特征的有效秩和特征多样性，降低不同特征之间的相关性，同时增强模型对不同尺度退化和局部结构的适应能力。
'''
'''
从SCI论文角度，我自己的定义：（供群里小伙伴参考！）
AERLA首先利用线性注意力捕获全局依赖关系，并从其输出特征中估计跨维冗余度与能量集中程度，作为有效秩退化的轻量代理。
随后，通过Rank-aware Router自适应分配不同尺度深度卷积分支的权重，
并动态调节局部特征补偿强度，从而由传统RELA中的固定秩增强转变为按需的自适应秩增强。
'''


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(2, 32, 128, 128).to(device)
    model = AERLA(dim=32).to(device)
    # 前向传播
    output = model(input)
    print('ECCV 2026 顶会二次创新模块-AERLA input_size:', input.size())
    print('ECCV 2026 顶会二次创新模块-AERLA output_size:', output.size())
    print('有关RELA的二次创新代码，会更新在顶会顶刊二次创新改进交流群！')
    # AERLA 是ECCV 2026 RELA的二次创新模块,在二次创新交流群！
