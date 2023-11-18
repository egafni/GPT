from dataclasses import dataclass

import torch
from torch import nn

from gpt.model import NextToken


class Transformer(NextToken):
    @dataclass
    class Config:
        vocab_size: int
        block_size: int
        n_embd: int
        # n_hidden: int
        n_heads: int
        n_blocks: int

    def __init__(self, c: Config):
        super().__init__()
        self.c = c

        self.token_embedding_table = nn.Embedding(c.vocab_size, c.n_embd)
        self.position_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.Sequential(
            *(TransformerBlock(c.n_heads, c.n_embd, in_channels=c.n_embd if i == 0 else c.n_embd)
              for i in range(c.n_blocks)
              ))
        self.project = nn.Linear(c.n_embd, c.vocab_size)

        self.register_buffer('positions', torch.arange(c.block_size))

    def forward(self, x, targets=None) -> (torch.Tensor, torch.Tensor):
        B, T, C, H, V = x.shape[0], self.c.block_size, self.c.n_embd, self.c.n_embd, self.c.vocab_size
        assert x.shape == (B, T)

        emb = self.token_embedding_table(x)  # B,T,C
        assert emb.shape == (B, T, C)
        assert not x.isnan().any()

        pos = self.position_embedding_table(self.positions)  # T,C
        assert pos.shape == (T, C)
        assert not x.isnan().any()

        x = emb + pos  # B,T,C
        assert x.shape == (B, T, C)
        assert not x.isnan().any()

        x = self.blocks(x)  # B,T,H

        x = self.project(x)  # B,T,V
        assert x.shape == (B, T, V)

        logits = x

        if targets is None:
            loss = None
        else:
            logits2 = logits.view(B * T, V)
            targets2 = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits2, targets2)

        return logits, loss


class AttentionHead(nn.Module):
    def __init__(self, n_channels, head_size):
        super().__init__()
        C, H = n_channels, head_size

        self.H = H

        self.q = nn.Linear(C, H, bias=False)
        self.k = nn.Linear(C, H, bias=False)
        self.v = nn.Linear(C, H, bias=False)

        self.register_buffer('scale', torch.tensor(H ** -0.5))

    def forward(self, x):
        T = x.shape[1]  # time dim
        mask = ~torch.tril(torch.ones(T, T).to(torch.bool)).to(x.device)  # causal mask

        # forward
        q, k, v = self.q(x), self.k(x), self.v(x)  # B,T,C
        w = q @ k.transpose(-2, -1)  # B,T,C @ B,C,T = B,T,T
        w = torch.masked_fill(w, mask, value=float('-inf'))
        w = w * self.scale  # causal mask, normalize
        w = torch.softmax(w, dim=-1)
        assert w.isnan().sum() == 0
        x = w @ v  # B,T,T @ B,T,C = B,T,C
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, in_channels):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.in_channels = in_channels
        self.heads = nn.ModuleList([AttentionHead(n_channels=in_channels, head_size=head_size) for _ in range(n_heads)])

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, head_size, in_channels):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)
        self.mha = MultiHeadAttention(n_heads, head_size // n_heads, in_channels=in_channels)
        self.ln2 = nn.LayerNorm(head_size)
        self.ff = FeedForward(head_size)

    def forward(self, x):
        x = x+ self.mha(self.ln1(x))
        x = x+ self.ff(self.ln2(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, n_features * 4),
                                    nn.ReLU(),
                                    nn.Linear(n_features * 4, n_features))

    def forward(self, x):
        return self.layers(x)
