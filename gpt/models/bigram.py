import torch
from torch import nn

from gpt.models.model import NextToken


class BiGram(NextToken):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None) -> (torch.Tensor, torch.Tensor):
        logits = self.token_embedding_table(x)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits2 = logits.view(B * T, C)
            targets2 = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits2, targets2)

        return logits, loss