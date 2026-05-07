import torch
from torch import nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, dim, vocab_size, ctx_window, num_heads):
        self.dim= dim
        self.vocab_size = vocab_size
        self.ctx_window = ctx_window
        self.num_heads = num_heads
        
        
        
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.pos_enc = nn.Embedding(config.ctx_window, config.dim)
        
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.K = nn.Linear(config.dim, config.dim//config.num_heads)
        self.Q = nn.Linear(config.dim, config.dim//config.num_heads)
        self.V = nn.Linear(config.dim, config.dim//config.num_heads)
        

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(config) for _ in range(config.num_heads)])

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)   # before attention
        self.attention = MHA(config)
        self.ln2 = nn.LayerNorm(config.dim)   # before FC
        self.fc = FC(config)
    def forward(self, x):
        x = x + self.attention(self.ln1(x))   # residual + attention
        x = x + self.fc(self.ln2(x))          # residual + FC
        return x

class FC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim),
            nn.GELU(),
            
        )
        