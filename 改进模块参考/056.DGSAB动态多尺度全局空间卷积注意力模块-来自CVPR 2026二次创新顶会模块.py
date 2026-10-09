import torch
import torch.nn as nn
import torch.nn.functional as F
#论文：LRDUN: A Low-Rank Deep Unfolding Network for Efficient Spectral Compressive Imaging
#看Ai缝合怪b站视频：2026.10.1 更新的视频
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type="ln"):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        self.norm = nn.LayerNorm(dim) if norm_type == "ln" else nn.GroupNorm(dim, dim)

    def forward(self, x):
        if self.norm_type == "ln":
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)
        return self.fn(x)


class FFN(nn.Module):
    def __init__(self, dim, mult=2.66):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class DynamicMultiScaleAttention(nn.Module):
    """动态选择 5×5 / 7×7 / 11×11 感受野"""

    def __init__(self, dim, kernels=(5, 7, 11), router_ratio=4):
        super().__init__()
        self.pre = nn.Conv2d(dim, dim, 1, bias=False)

        self.branches = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in kernels
        ])

        hidden = max(dim // router_ratio, 4)
        self.router = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, len(kernels), 1),
        )

        self.act = nn.GELU()
        self.apply(init_weights)

    def forward(self, x):
        feat = self.pre(x)

        # [B, S, C, H, W]
        multi_scale = torch.stack(
            [branch(feat) for branch in self.branches],
            dim=1
        )

        # [B, S, 1, H, W]
        weights = torch.softmax(self.router(x), dim=1).unsqueeze(2)

        return self.act((multi_scale * weights).sum(dim=1))


class PooledGlobalAttention(nn.Module):
    """低分辨率全局 Self-Attention"""

    def __init__(self, dim, heads=1, pool_size=8):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.pool_size = pool_size
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.apply(init_weights)

    def forward(self, x):
        b, c, h, w = x.shape
        ph, pw = min(h, self.pool_size), min(w, self.pool_size)

        feat = F.adaptive_avg_pool2d(x, (ph, pw))
        q, k, v = self.to_qkv(feat).chunk(3, dim=1)

        n = ph * pw

        def reshape(t):
            return (
                t.flatten(2)
                .transpose(1, 2)
                .reshape(b, n, self.heads, self.dim_head)
                .transpose(1, 2)
            )

        q, k, v = map(reshape, (q, k, v))

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = attn @ v

        out = (
            out.transpose(1, 2)
            .reshape(b, n, c)
            .transpose(1, 2)
            .reshape(b, c, ph, pw)
        )

        out = self.proj(out)

        return F.interpolate(
            out,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )


class DynamicGlobalCMB(nn.Module):
    """
    Dynamic Multi-Scale + Global Context CMB
    """

    def __init__(self, dim, kernels=(5, 7, 11), heads=1, pool_size=8):
        super().__init__()

        self.local = DynamicMultiScaleAttention(dim, kernels)
        self.global_attn = PooledGlobalAttention(dim, heads, pool_size)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )

        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

        self.local_scale = nn.Parameter(torch.tensor(1.0))
        self.global_scale = nn.Parameter(torch.tensor(1.0))

        self.apply(init_weights)

    def forward(self, x):
        local = self.local_scale * self.local(x)
        global_feat = self.global_scale * self.global_attn(x)

        gate = self.fuse(torch.cat([local, global_feat], dim=1))
        return self.to_out(gate * self.to_v(x))


class DGSAB(nn.Module):
    """
    Dynamic Multi-Scale Global Spatial Conv Attention Block
    """
    def __init__(self, dim, heads=4, kernels=(5, 7, 11), global_pool=8):
        super().__init__()

        self.attn = PreNorm(
            dim,
            DynamicGlobalCMB(
                dim,
                kernels=kernels,
                heads=heads,
                pool_size=global_pool
            )
        )

        self.ffn = PreNorm(
            dim,
            FFN(dim),
            norm_type="gn"
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

#DGSAB: Dynamic Multi-Scale Global Spatial Conv Attention Block
#可以直接拿去发论文，作为CCF-A/B/C，或是SCI一区/二区/三区/四区 的一种即插即用涨点创新模块

'''
原始SCAB 通过固定的11×11 Depthwise Conv 扩大感受野，并用卷积式注意力增强长程空间依赖，这种设计简单高效，
但也存在两个明显局限：
一方面，固定尺度的 11×11 卷积无法针对不同区域、不同结构自适应调整感受范围；
另一方面，即便 11×11 已经比普通卷积看得更远，它本质上仍属于局部建模，难以获得真正的全局上下文。

因此，我们设计了动态多尺度全局空间卷积注意力模块，在局部分支中引入 5×5、7×7、11×11 多尺度 Depthwise Conv，
并利用输入相关的动态路由为不同空间位置自适应分配尺度权重，使网络能够根据纹理、边缘和结构复杂度灵活调整感受野；
同时增加轻量级全局上下文分支，通过低分辨率全局注意力补充跨区域依赖，最终将动态局部信息与全局语义进行融合，
在尽量保持SCAB原有高效性的基础上，弥补其固定感受野和局部建模能力不足的问题。
'''

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(2, 32, 128, 128).to(device)
    model = DGSAB(
        dim=32,
        heads=4,  #可以灵活去设置修改
        kernels=(5, 7, 11), #可以灵活去设置修改
        global_pool=8,  #可以灵活去设置修改
    ).to(device)
    # 前向传播
    output = model(input)
    print('CVPR 2026 顶会二次创新模块-DGSAB input_size:', input.size())
    print('CVPR 2026 顶会二次创新模块-DGSAB output_size:', output.size())
    print('有关SCAB的二次创新代码，会更新在顶会顶刊二次创新改进交流群！')
    # DGSAB 是CVPR 2026 SCAB的二次创新模块,在二次创新交流群！
