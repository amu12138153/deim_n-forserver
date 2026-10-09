# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT  # pip install torch_dct
from einops import rearrange
#关于HFP模块内容介绍，可以去看Ai缝合怪b站视频：2025.7.17 更新的视频
class DctSpatialInteraction(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio,
                 isdct=True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct  # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                *[nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)]
            )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')  # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctChannelInteraction(nn.Module):
    def __init__(self,
                 in_channels,
                 patch,
                 ratio,
                 isdct=True
                 ):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.channel2x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:  # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))  # 2025 03 15 szc
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)  # 2025 03 15 szc
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

# High Frequency Perception Module HFP
# ------------------------------------------------------------------#
class HFP(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 patch=(8, 8),
                 isdct=True):
        super(HFP, self).__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
              nn.GroupNorm(32, in_channels)]
        )

    def forward(self, x):
        spatial = self.spatial(x)  # output of spatial path
        channel = self.channel(x)  # output of channel path
        return self.out(spatial + channel)
from einops.layers.torch import Rearrange
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
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
'''二次创新模块:HLPFM高低频感知融合模块'''
class HLPFM(nn.Module):
    def __init__(self, dim):
        super(HLPFM, self).__init__()
        self.HFP = HFP(dim)
        self.pa = PixelAttention(dim)
        self.conv =Conv(dim, dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        pattn1 = self.HFP(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        # 1. pattn2 可以看成高频特征权重；(1 - pattn2) 可以看成低频权重；
        # 2. pattn2 * x + (1 - pattn2) * y 和 pattn2 * y + (1 - pattn2) * x 这两种加权都可以灵活的去调整加权的对象；跑跑实验看效果
        # 3. HFP 模块的  频率范围ratio=(0.25, 0.25), 自己也可以去做一下在自己数据集上的消融实验；在这篇论文里面默认0.25是最好的；这个超参数到时候自己也可以作为自己论文的消融实验，讲故事丰富论文工作量
        result = self.conv(result)
        return result

if __name__ == '__main__':

    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 64, 128, 128)
    input2 = torch.randn(1, 64, 128, 128)
    # AAAI2025 HFP模块的二次创新，HLPFM在我的二次创新模块交流群，可以直接去发小论文！
    # 创建 HLPFM 模块
    HLPFM = HLPFM(dim=64)
    # 将输入图像传入HLPFM模块进行处理
    output = HLPFM(input1,input2)
    print('顶会顶刊二次创新模块永久更新-HLPFM_input_size:', input1.size())
    print('顶会顶刊二次创新模块永久更新-HLPFM_output_size:', output.size())


