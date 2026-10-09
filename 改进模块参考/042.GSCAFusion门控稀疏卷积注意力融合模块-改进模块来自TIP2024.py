import torch
import torch.nn as nn
from einops import rearrange
'''

二次创新模块： GSCAFusion
基于CGAFusion（2024 TIP顶刊）模块的框架，同时利用ACM MM 2025中的利用EPGO动态稀疏Top-K机制改进CAFM（2024 SCI 二区）模块

CAFM（2024 SCI 二区）:所提出的卷积和注意力特征融合模块。它由局部和全局分支组成。
    在局部分支中，采用卷积和通道洗牌进行局部特征提取。
    在全局分支中，注意力机制用于对长程特征依赖关系进行建模。

CGAFusion（2024 TIP顶刊）:我们提出了一种新的注意机制，可以强调用特征编码的更多有用的信息，以有效地提高性能。
    此外，还提出了一种基于CGA的混合融合方案，可以有效地将编码器部分的低级特征与相应的高级特征进行融合。

强强联手：GSCAFusion用于低级特征与高级特征校准/融合模块 （冲二，三，保四）
适用于：图像去噪，图像增强，目标检测，语义分割，实例分割，图像恢复，暗光增强等所有CV2维任务
'''
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class CSAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CSAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        B,C,H,W = x.shape
        X=x

        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
#-------利用动态稀疏Top-K机制，学习重要信息，引导注意力关注关键区域。
#       实现内容感知的注意力引导，提升上下文建模能力。
        _, _, C, _ = q.shape
        gate_output = self.gate(X)
        if torch.isnan(gate_output).any():
            gate_output = torch.nan_to_num(gate_output, nan=0.0)  # 将 NaN 转换为 0

        dynamic_k = int(C * gate_output.view(B, -1).mean())
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask = torch.zeros(B, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
#-------------------------------------
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
        return output

class GSCAFusion(nn.Module): #Gated Sparse Convolution-Attention Fusion Module
    def __init__(self, dim):
        super(GSCAFusion, self).__init__()
        self.CSAFM = CSAFM(dim)
        self.PixelAttention = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        pattn1 = self.CSAFM(initial)
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

if __name__ == '__main__':
    block = GSCAFusion(32)
    input1 = torch.rand(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)
    output = block(input1, input2)
    print('input1_size:', input1.size())
    print('input2_size:', input2.size())
    print('output_size:', output.size())
