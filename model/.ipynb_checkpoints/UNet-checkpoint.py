# The layers and models related to the noise2noise prediction


import torch
from einops import rearrange



# Helper classes

# Residual connections
class Residual(torch.nn.Module): 
    
    def __init__(self, fn): 
        super(Residual, self).__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs): 
        return self.fn(x, *args, **kwargs) + x

# up and down sampling from convolution layers
def Upsample(dim): return torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)
def Downsample(dim): return torch.nn.Conv2d(dim, dim, 4, 2, 1)



# Group Normalization
class GNorm(torch.nn.Module): 

    def __init__(self, dim, fn): 
        super(GNorm, self).__init__()
        self.dim = dim
        self.fn = fn
        self.norm = torch.nn.GroupNorm(1, dim)

    def forward(self, x): 
        return self.fn(self.norm(x))
        
        

# Block Layer
class Block(torch.nn.Module): 
    
    def __init__(self, dim, output_dim, groups=8): 
        super(Block, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.groups = groups

        # projection, normalize and activation
        self.proj = torch.nn.Conv2d(dim, output_dim, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, output_dim)
        self.act = torch.nn.SiLU()

    def forward(self, x, scale_shift=None): 
        x = self.proj(x)
        x = self.norm(x)
        
        if exists(scale_shift): 
            scale, shift = scale_shift
            x = (x * scale) + shift
        
        return self.act(x)

# ResNet: Noise2Noise
class ResNetBlock(torch.nn.Module): 

    def __init__(self, dim, output_dim, *, time_emb_dim=None, groups=8):
        super(ResNetBlock, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.time_emb_dim = time_emb_dim
        self.groups = groups

        self.mlp = torch.nn.Sequential(torch.nn.SiLU(), 
                                       torch.nn.Linear(time_emb_dim, output_dim)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, output_dim, groups=groups)
        self.block2 = Block(output_dim, output_dim, groups=groups)
        self.res_conv = torch.nn.Conv2d(dim, output_dim, 1) if dim != output_dim else torch.nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, 'b c -> b c 1 1') + h
        
        h = self.block2(h)
        return self.res_conv(x) + h


# ConvNeXT: Noise2Noise

class ConvNextBlock(torch.nn.Module): 
    def __init__(self, dim, output_dim, *, time_emb_dim=None, mult=2, norm=True): 
        super(ConvNextBlock, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.time_emb_dim = time_emb_dim
        
        self.mlp = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None
        
        self.ds_conv = torch.nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = torch.nn.Sequential(torch.nn.GroupNorm(1, dim) if norm else torch.nn.Identity(), 
                                       torch.nn.Conv2d(dim, output_dim * mult, 3, padding=1), 
                                       torch.nn.GELU(),
                                       torch.nn.GroupNorm(1, output_dim * mult), 
                                       torch.nn.Conv2d(output_dim * mult, output_dim, 3, padding=1))
        self.res_conv = torch.nn.Conv2d(dim, output_dim, 1) if dim != output_dim else torch.nn.Identity()
        
        
    def forward(self, x, time_emb=None): 
        h = self.ds_conv(x)
        
        if exists(self.mlp) and exists(time_emb): 
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, 'b c -> b c 1 1')

        h = self.net(h)
        # print(h.shape)
        return self.res_conv(x) + h




# UNet 
# # UNet
# # Should use ResNet for the starter

class UNet(torch.nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = torch.nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(in_out)
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResNetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = torch.nn.Sequential(
                TimeStepEmbedding(dim),
                torch.nn.Linear(dim, time_dim),
                torch.nn.GELU(),
                torch.nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(GNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(GNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(GNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = torch.nn.Sequential(
            block_klass(dim, dim), torch.nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # print('mid x: ', x.shape)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            # print('x: ', x.shape, 'h: ', h[-1].shape)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

        

# UNet++



# Other flavors of UNet

