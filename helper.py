import torch
from torch import nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, dim, vocab_size, ctx_window, num_heads, num_blocks):
        self.dim= dim
        self.vocab_size = vocab_size
        self.ctx_window = ctx_window
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        
        
        
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.dim)
        self.wpe = nn.Embedding(config.ctx_window, config.dim)
        
        # self.drop = nn.Drop(0.1, inplace= False)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.num_blocks)])
        self.head = nn.Linear(config.dim, config.vocab_size)
        self.ln_f = nn.LayerNorm(config.dim)
        self.apply(self._init_weights)

    
    
    def forward(self, x):
        emb = self.wte(x)
        pos = self.wpe(torch.arange(x.shape[1]))
        x = emb +pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x =self.head(x)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.K = nn.Linear(config.dim, config.dim//config.num_heads)
        self.Q = nn.Linear(config.dim, config.dim//config.num_heads)
        self.V = nn.Linear(config.dim, config.dim//config.num_heads)
        self.scaler = config.dim**0.5
        self.apply(self._init_weights)

        
    def forward(self,x):
        tril = torch.tril(torch.ones(x.shape[1], x.shape[1]))

        
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        scores = (q @ k.transpose(-2, -1)) / self.scaler
        tril = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device))
        scores = scores.masked_fill(tril == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

        

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.dim, config.dim) 
        self.apply(self._init_weights)
    def forward(self, x):
        heads_out = [sa(x) for sa in self.heads]
        x = torch.cat(heads_out, dim=-1)
        x = self.proj(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

        

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)   # before attention
        self.attention = MHA(config)
        self.ln2 = nn.LayerNorm(config.dim)   # before FC
        self.fc = FC(config)
        self.apply(self._init_weights)
    def forward(self, x):
        x = x + self.attention(self.ln1(x))   # residual + attention
        x = x + self.fc(self.ln2(x))          # residual + FC
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

class FC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim)
            
        )
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.fc(x)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

