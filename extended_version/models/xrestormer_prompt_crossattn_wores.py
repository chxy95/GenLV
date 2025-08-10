import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numbers

from inspect import isfunction
from einops import rearrange
from basicsr.archs.arch_util import ResidualBlockNoBN, make_layer

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    # col_pad = torch.zeros((b, l, 1), **to(x))
    col_pad = torch.zeros((b, l, 1)).to(x.device).to(x.dtype)
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    # flat_pad = torch.zeros((b, m - l), **to(x))
    flat_pad = torch.zeros((b, m - l)).to(x.device).to(x.dtype)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Overlapping Cross-Attention Block (OCAB)
class OCAB(nn.Module):
    def __init__(self, dim, window_size, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        overlap_ratio = 0.5
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class SpatCrossAttn(nn.Module):
    def __init__(self, query_dim, prompt_dim=None, heads=8, dropout=0.):
        super().__init__()
        inner_dim = query_dim # same dimension for attention calculation.
        prompt_dim = default(prompt_dim, query_dim)

        self.scale = (inner_dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(prompt_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(prompt_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, prompt_K=None, prompt_V=None):
        h = self.heads

        q = self.to_q(x)

        if prompt_K is None or prompt_V is None:
            k = self.to_k(x)
            v = self.to_v(x)
        else:
            k = self.to_k(prompt_K)
            v = self.to_v(prompt_V)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, ffn_expansion_factor, bias, LayerNorm_type, spatial_dim_head=16):
        super(TransformerBlock, self).__init__()

        self.channel_attn = MDTA(dim, num_heads, bias)
        self.spatial_attn = OCAB(dim, window_size, num_heads, spatial_dim_head, bias)
    
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

##########################################################################
class SpatCrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SpatCrossAttnBlock, self).__init__()

        self.channel_attn = MDTA(dim, num_heads, bias)
        self.spatial_attn = SpatCrossAttn(query_dim=dim, heads=num_heads)
        self.cross_attn = SpatCrossAttn(query_dim=dim, prompt_dim=dim, heads=num_heads)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, prompt_lq, prompt_gt):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))

        _, _, h, _ = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.cross_attn(self.norm4(x), prompt_lq, prompt_gt)
        x = self.norm5(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h).contiguous()
        x = x + self.spatial_ffn(x)

        return x

##########################################################################
class LatenBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(LatenBlock, self).__init__()

        self.channel_attn = MDTA(dim, num_heads, bias)
        self.spatial_attn = SpatCrossAttn(query_dim=dim, heads=num_heads)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))

        _, _, h, _ = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x + self.spatial_attn(self.norm3(x))
        x = self.norm4(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h).contiguous()
        x = x + self.spatial_ffn(x)

        return x
    
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class Single_Base_Encoder_x3(nn.Module):
    def __init__(self, in_channel=3, prompt_dim=512, nf=32): # nf=64
        super(Single_Base_Encoder_x3, self).__init__()

        self.conv_first = nn.Conv2d(in_channel, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1) # ori * ori * nf
        self.norm = nn.LayerNorm(nf)

        self.fea_level1 = make_layer(ResidualBlockNoBN, 4, num_feat=nf)
        self.pixel_unshuffle1 = nn.PixelUnshuffle(2) # ori//2 * ori//2 * nf*4
        self.channel_reduction1 = nn.Conv2d(4 * nf, 2 * nf, 1) # ori//2 * ori//2 * nf*2
        self.norm1 = nn.LayerNorm(2 * nf)

        self.fea_level2 = make_layer(ResidualBlockNoBN, 4, num_feat=2 * nf)
        self.pixel_unshuffle2 = nn.PixelUnshuffle(2) # ori//4 * ori//4 * nf*2*4
        self.channel_reduction2 = nn.Conv2d(4 * 2 * nf, 4 * nf, 1) # ori//4 * ori//4 * nf*4
        self.norm2 = nn.LayerNorm(4 * nf)

        self.fea_level3 = make_layer(ResidualBlockNoBN, 4, num_feat=4 * nf)
        self.pixel_unshuffle3 = nn.PixelUnshuffle(2) # ori//8 * ori//8 * nf*4*4
        self.channel_reduction3 = nn.Conv2d(4 * 4 * nf, 8 * nf, 1) # ori//4 * ori//4 * nf*8
        self.norm3 = nn.LayerNorm(8 * nf)
        
        self.conv_lr = nn.Conv2d(8 * nf, prompt_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(prompt_dim, prompt_dim, 1)
        self.norm_out = nn.LayerNorm(prompt_dim)

        # activation function
        self.gelu = nn.GELU()

    def forward(self, x):
        fea = self.conv_hr(self.gelu(self.conv_first(x)))
        _, _, h, _ = fea.shape
        fea = self.norm(rearrange(fea, 'b c h w -> b (h w) c'))
        fea = rearrange(fea, 'b (h w) c -> b c h w', h=h).contiguous()

        fea_l1 = self.gelu(self.fea_level1(fea))
        fea_l1 = self.gelu(self.channel_reduction1(self.pixel_unshuffle1(fea_l1)))
        _, _, h1, _ = fea_l1.shape
        fea_l1 = self.norm1(rearrange(fea_l1, 'b c h w -> b (h w) c'))
        fea_l1 = rearrange(fea_l1, 'b (h w) c -> b c h w', h=h1).contiguous()

        fea_l2 = self.gelu(self.fea_level2(fea_l1))
        fea_l2 = self.gelu(self.channel_reduction2(self.pixel_unshuffle2(fea_l2)))
        _, _, h2, _ = fea_l2.shape
        fea_l2 = self.norm2(rearrange(fea_l2, 'b c h w -> b (h w) c'))
        fea_l2 = rearrange(fea_l2, 'b (h w) c -> b c h w', h=h2).contiguous()

        fea_l3 = self.gelu(self.fea_level3(fea_l2))
        fea_l3 = self.gelu(self.channel_reduction3(self.pixel_unshuffle3(fea_l3)))
        _, _, h3, _ = fea_l3.shape
        fea_l3 = self.norm3(rearrange(fea_l3, 'b c h w -> b (h w) c'))
        fea_l3 = rearrange(fea_l3, 'b (h w) c -> b c h w', h=h3).contiguous()
        
        out = self.conv_last(self.gelu(self.conv_lr(fea_l3)))
        out = self.norm_out(rearrange(out, 'b c h w -> b (h w) c'))
        return out
    
    
class Prompt_Unified_KV_Encoder_x3(nn.Module):
    def __init__(self, in_channel=3, prompt_dim=512):
        super(Prompt_Unified_KV_Encoder_x3, self).__init__()
        print('Prompt Encoder: Prompt_Unified_KV_Encoder_x3')

        self.prompt_embedding = Single_Base_Encoder_x3(in_channel, prompt_dim)

    def forward(self, input, target):
        prompt_K = self.prompt_embedding(input)
        prompt_V = self.prompt_embedding(target)
        return prompt_K, prompt_V

    # def forward(self, x):
    #     prompt_K = self.prompt_embedding(x[0])
    #     prompt_V = self.prompt_embedding(x[1])
    #     return prompt_K, prompt_V
    
##########################################################################
##---------- XRestormer_Prompt_SpatCrossAttn -----------------------
class XRestormer_Prompt_SpatCrossAttn(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        window_size = 8, 
        num_blocks = [2,4,4,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
    ):

        super(XRestormer_Prompt_SpatCrossAttn, self).__init__()

        self.Prompt_KV_Encoder = Prompt_Unified_KV_Encoder_x3(prompt_dim=int(dim*2**3))

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, window_size=window_size, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size=window_size, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent_blocks = nn.ModuleList([SpatCrossAttnBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 
                                     for _ in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size=window_size, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.L1_loss = nn.L1Loss()

    def forward_loss_pix(self, pred_target_img, target_img):
        loss = self.L1_loss(pred_target_img, target_img)
        return loss

    def forward(self, imgs, visual_tokens=None, input_is_list=False, mask_ratio=None):
        loss = {}

        if not input_is_list:
            # split input images
            B, S, C, H, W = imgs.shape
            prompt_input = imgs[:, 0, :, :, :]
            prompt_target = imgs[:, 1, :, :, :]
            input_img = imgs[:, 2, :, :, :]
            target_img = imgs[:, 3, :, :, :]
        else:
            prompt_input = imgs[0]
            prompt_target = imgs[1]
            input_img = imgs[2]
            target_img = imgs[3]
        
        prompt_lq, prompt_gt = self.Prompt_KV_Encoder(prompt_input, prompt_target)
        # prompt_lq = torch.rand(1, 1024, 384)
        # prompt_gt = torch.rand(1, 1024, 384)

        inp_enc_level1 = self.patch_embed(input_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        latent = self.down3_4(out_enc_level3)

        for block in self.latent_blocks:
            latent = block(latent, prompt_lq, prompt_gt)    
                             
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        pred_target_img = self.output(out_dec_level1)

        if visual_tokens is not None:
            target_img = visual_tokens[:, 3, :, :, :]
            loss['pix_loss'] = self.forward_loss_pix(pred_target_img, target_img)
        return loss, pred_target_img, None


##########################################################################
##---------- XRestormer -----------------------
class XRestormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        window_size = 8, 
        num_blocks = [2,4,4,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
    ):

        super(XRestormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, window_size=window_size, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size=window_size, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent_blocks = nn.ModuleList([LatenBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 
                                     for _ in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size=window_size, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size=window_size, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.L1_loss = nn.L1Loss()

    def forward_loss_pix(self, pred_target_img, target_img):
        loss = self.L1_loss(pred_target_img, target_img)
        return loss

    def forward(self, imgs, visual_tokens=None, input_is_list=False, mask_ratio=None):
        loss = {}

        if not input_is_list:
            # split input images
            B, S, C, H, W = imgs.shape
            input_img = imgs[:, 0, :, :, :]
            target_img = imgs[:, 1, :, :, :]
        else:
            input_img = imgs[0]
            target_img = imgs[1]

        inp_enc_level1 = self.patch_embed(input_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        latent = self.down3_4(out_enc_level3)

        for block in self.latent_blocks:
            latent = block(latent)    
                             
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        pred_target_img = self.output(out_dec_level1)

        if visual_tokens is not None:
            target_img = visual_tokens[:, 1, :, :, :]
            loss['pix_loss'] = self.forward_loss_pix(pred_target_img, target_img)
        return loss, pred_target_img, None
    
    
def xrestormer_base(**kwargs): 
    model = XRestormer(dim=48, 
                       num_blocks=[2,4,4,4], 
                       num_refinement_blocks=4)
    return model

def xrestormer_huge(**kwargs): # FLOPs = 1141.18G, Params = 211.11M, max mem: 76000M
    model = XRestormer(dim=80,
                       num_blocks = [6,8,8,12], 
                       num_refinement_blocks=8)
    return model

def xrestormer_prompt_crossattn_base(**kwargs): # FLOPs = 296.55G, Params = 38.77M, max mem: 73000M
    model = XRestormer_Prompt_SpatCrossAttn(dim=48, 
                                            num_blocks=[2,4,4,4], 
                                            num_refinement_blocks=4)
    return model

def xrestormer_prompt_crossattn_large(**kwargs): # FLOPs = 603.59G, Params = 100.48M, max mem: 77000M
    model = XRestormer_Prompt_SpatCrossAttn(dim=64, 
                                            num_blocks = [4,6,6,8], 
                                            num_refinement_blocks=6)
    return model

def xrestormer_prompt_crossattn_huge(**kwargs): # FLOPs = 1141.18G, Params = 211.11M, max mem: 76000M
    model = XRestormer_Prompt_SpatCrossAttn(dim=80,
                                            num_blocks = [6,8,8,12], 
                                            num_refinement_blocks=8)
    return model

def xrestormer_prompt_crossattn_giant(**kwargs): # FLOPs = 2175.78G, Params = 401.57M, max mem: 74000M
    model = XRestormer_Prompt_SpatCrossAttn(dim=96, 
                                            num_blocks = [8,12,12,16], 
                                            num_refinement_blocks=12)
    return model

if __name__ == "__main__":
    # model = xrestormer_prompt_crossattn_base()
    # x = torch.randn((1, 4, 3, 256, 256)) # FLOPs = 201.58488576G, Params = 32.696256M
    
    # model = XRestormer()
    # x = torch.randn((1, 2, 3, 256, 256)) # FLOPs = 165.8966016G, Params = 27.56496M
    
    # pred = model(x)
    # _, pred, _ = model(x)
    # print(pred.shape)
    
    model = Prompt_Unified_KV_Encoder_x3(prompt_dim=384)
    t = torch.randn((1, 3, 256, 256)) # FLOPs = 33.2660736G, Params = 2.767392M
    x = (t,t)
    pred = model(x)
    
    from thop import profile
    # flops, params = profile(model, inputs=(x,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    
    flops, params = profile(model, inputs=(x,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')