import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import to_2tuple
import torch_dct  # pip install torch_dct -i https://pypi.tuna.tsinghua.edu.cn/simple
from torch.nn import functional as F
#看Ai缝合怪b站视频：2025.6.13更新的视频

class FEFM(nn.Module):
    def __init__(self, in_channels,num_heads=4,init_lambda=0.5):
        super(FEFM, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1))
        self.conv_Fn_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_Fn_k = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_Fr_q = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(in_channels*2)
        # self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=1,groups=in_channels * 2, bias=True)
        self.init_lambda = init_lambda
        self.patch_size = 8
        self.num_heads = num_heads

    def forward(self, input1,input2):
        b,c,h,w = input1.shape
        F_Q =  self.conv_Fr_q(input1)
        F_K =  self.conv_Fn_k(input2)
        F_V =  self.conv_Fn_v(input2)


        q_patch = rearrange(F_Q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size, patch2=self.patch_size)
        k_patch = rearrange(F_K, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        fft_Q =  torch_dct.dct_2d(q_patch.float())
        fft_K =  torch_dct.dct_2d(k_patch.float())

        F1 = fft_Q @ fft_K #torch.matmul(fft_Q ,fft_K)

        F1 = rearrange(F1, 'b c h w patch1 patch2 -> b  c (h w patch1 patch2)', patch1=self.patch_size, patch2=self.patch_size)
        Fq = rearrange(fft_Q, 'b c h w patch1 patch2 -> b  c (h w patch1 patch2)',patch1=self.patch_size,patch2=self.patch_size)
        Fk = rearrange(fft_K, 'b c h w patch1 patch2 -> b  c (h w patch1 patch2)', patch1=self.patch_size,patch2=self.patch_size)

        attn = (Fq @  Fk.transpose(-2, -1))*self.temperature

        attn = attn.softmax(dim=-1)
        Fcfr =  attn @ F1
        Fcfr = Fcfr.reshape(b,c,h // self.patch_size ,w // self.patch_size,self.patch_size,self.patch_size)
        Fcfr_out =  torch_dct.idct_2d(Fcfr)
        Fcfrout = rearrange(Fcfr_out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size)

        #DFR操作
        out = Fcfrout @ F_V
        Fdfr = torch.sub(F_V,out * self.init_lambda )
        out =  Fcfrout @ F_Q + Fdfr
        return out

#二次创新CFEM，交叉融合增强模块，冲SCI三区和四区，CCF-B/C
'''
#论文地址： https://arxiv.org/abs/2502.20272
来自CVPR 2025顶会 低光图像增强任务
即插即用模块: LCA 轻量交叉注意力模块
卷积和注意力特征融合模块：Cross-Attention       
背景：  
LCA（Lighten Cross-Attention）模块的作用:
是增强亮度信息和颜色信息之间的特征交互，以提高低光图像的增强效果。
它采用交叉注意力机制，让亮度分支和颜色分支相互提供信息，从而避免单独处理导致的亮度偏差或颜色失真。
通过计算亮度特征对颜色特征的注意力分布，以及颜色特征对亮度特征的关注程度，该模块能够更精准地调整光照强度，
并优化颜色一致性，使增强后的图像更加自然。

IEL（Intensity Enhance Layer）模块的作用:
是提升图像的亮度表现，同时减少光照不均和低光区域的噪声。它基于 Retinex 理论，将图像的亮度信息和反射信息进行分离，
并通过非线性变换增强亮度特征，使低光区域的细节得以保留，同时避免过度增强导致的伪影或光晕。
通过自适应调整亮度，该模块可以有效改善低光场景下的图像质量，使图像整体亮度更均衡、细节更丰富。

适用于：低光图像增强，图像去噪，图像恢复，图像去雨雪/雾/模糊，目标检测，图像分割，等所有CV任务通用的即插即用模块
'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# Intensity Enhancement Layer
class IEL(nn.Module): # 强度增强层
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
class CFEM(nn.Module):
    def __init__(self, in_channels,init_lambda=0.5):
        super(CFEM, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1))
        self.conv_Fn_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_Fn_k = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_Fr_q = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.init_lambda = init_lambda #init_lambda是平衡因子，可以自己设定
        self.cam = CrossAttention(in_channels)
        self.iel = IEL(in_channels)
        self.norm = LayerNorm(in_channels)

    def forward(self, input1,input2):
        F_Q =  self.conv_Fr_q(input1)
        F_K =  self.conv_Fn_k(input2)
        F_V =  self.conv_Fn_v(input2)
        # 交叉注意力
        att = self.cam(F_Q,F_K)
        #DFR操作
        out = att @ F_V
        Fdfr = torch.sub(F_V,out * self.init_lambda)
        out =  att @ F_Q + Fdfr

        #特征增强操作➕残差连接
        out = self.iel(self.norm(out)) + input1
        # 最后进行特征增强操作，在这里然后再进行残差操作，
        # 防止特征信息丢失，选择input1或是input2都可以
        # 举个例子：input1，input2分别代表高频和低频特征，如果想要保留更多高频信息，就使用input1高频信息，反之。或是使用input1+input2也行，可以多去跑跑使用验证。
        return out
# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 创建 FEFM 模块
    fefm = FEFM(in_channels=32)
    # 将输入图像传入 FEFM 模块进行处理
    output = fefm(input1,input2)
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-FEFM_input_size:', input1.size())
    print('Ai缝合即插即用模块永久更新-FEFM_output_size:', output.size())

    # CVPR2025 FEFM模块的二次创新，CFEM在我的二次创新模块交流群，可以直接去发小论文！
    # 创建 CFEM 模块
    fdam = CFEM(in_channels=32)
    # 将输入图像传入CFEM模块进行处理
    output = fdam(input1,input2)
    print('顶会顶刊二次创新模块永久更新-CFEM_input_size:', input1.size())
    print('顶会顶刊二次创新模块永久更新-CFEM_output_size:', output.size())

