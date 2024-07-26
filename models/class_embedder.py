import math
import torch 
import torch.nn as nn

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim=512, num_classes=1000, cond_drop_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        init_std = math.sqrt(1 / embed_dim / 3)
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)
        nn.init.trunc_normal_(self.embedding.weight.data, mean=0, std=init_std)
        self.cond_drop_rate = cond_drop_rate
        
    def forward(self, x):
        b = x.size(0)
        if self.cond_drop_rate > 0:
            x = torch.where(torch.rand(b, device=x.device) < self.cond_drop_rate, self.num_classes, x)
        c = self.embedding(x)
        return c