# attention layers


import torch
from einops import rearrange



# Original Attention (in the Conv Net Context)
class Attention(torch.nn.Module): 

    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.dim = dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False) # compute qkv at the same timne
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x): 
        b, c, h, w = x.shape # batch, channel, height, width
        
        qkv = self.to_qkv(x).chunk(3, dim=1) # (q, k, v) tuple
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1) # batch x heads x (h x w) x (h x w)
        # attention is pixel to pixel weight

        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        return self.to_out(out)


# Linear Attention
class LinearAttention(torch.nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=True)
        self.to_out = torch.nn.Sequential(torch.nn.Conv2d(hidden_dim, dim, 1), 
                                          torch.nn.GroupNorm(1, dim))

    def forward(self, x): 
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-2)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, y=w)
        return self.to_out(out)