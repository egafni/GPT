from torch import nn

from gpt.models.model import NextToken


class WaveNet(NextToken):
    """Doesn't work"""

    def __init__(self, vocab_size, n_embd, n_hidden):
        super().__init__()
        self.vocab_size = vocab_size
        H = n_hidden
        act = nn.Tanh
        self.layers = nn.Sequential(
            nn.Embedding(vocab_size, n_embd),
            # layer 1
            FlattenConsecutive(2),
            nn.Linear(n_embd * 2, H),
            Permute((0, 2, 1)),  # B, T, C -> B, C, T
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            Permute((0, 2, 1)),  # B, C, T -> B, T, C
            act(),

            # layer 2
            FlattenConsecutive(2),
            nn.Linear(H * 2, H),
            Permute((0, 2, 1)),  # B, T, C -> B, C, T
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            Permute((0, 2, 1)),  # B, C, T -> B, T, C
            act(),

            # layer 3
            FlattenConsecutive(2),
            nn.Linear(H * 2, H),
            Permute((0, 2, 1)),  # B, T, C -> B, C, T
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            Permute((0, 2, 1)),  # B, C, T -> B, T, C
            act(),
            nn.Linear(H, vocab_size),
        )

    def forward(self, x, targets=None):
        logits = self.layers(x)  # (B,T,C)
        return logits