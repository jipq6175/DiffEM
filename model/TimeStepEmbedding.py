# time step embedding 


import torch, math


# using the embeddings in the Transformer paper
class TimeStepEmbedding(torch.nn.Module): 

    def __init__(self, dim): 
        super(TimeStepEmbedding, self).__init__()
        assert dim % 2 == 0, 'Dimension has to be even number'
        self.dim = dim

    def forward(self, time): 
        # time is a tensor
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp( -torch.arange(half_dim, device=device) * embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings
    

# Using Fourier embedding
