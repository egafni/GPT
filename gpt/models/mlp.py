from torch import nn

from gpt.models.next_token import NextToken


# class MLP(NextToken):
#     def __init__(self, vocab_size, block_size, n_embd, n_hidden):
#         super().__init__()
#         self.vocab_size = vocab_size
#         H = n_hidden
#         act = nn.Tanh
#         self.layers = nn.Sequential(
#             nn.Embedding(vocab_size, n_embd),
#             nn.Flatten(),
#             # layer 1
#             nn.Linear(n_embd * block_size, H),
#             # Permute((0, 2, 1)),  # B, T, C -> B, C, T
#             nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
#             # Permute((0, 2, 1)),  # B, C, T -> B, T, C
#             act(),
#
#             # layer 2
#             nn.Linear(H, H),
#             # Permute((0, 2, 1)),  # B, T, C -> B, C, T
#             nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
#             # Permute((0, 2, 1)),  # B, C, T -> B, T, C
#             act(),
#
#             # layer 3
#             nn.Linear(H, H),
#             # Permute((0, 2, 1)),  # B, T, C -> B, C, T
#             nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
#             # Permute((0, 2, 1)),  # B, C, T -> B, T, C
#             act(),
#             nn.Linear(H, vocab_size),
#         )
#
#     def forward(self, x, targets=None):
#         logits = self.layers(x)  # (B,T,C)
#
#         # return logits[:, -1, :], None
#
#         if targets is None:
#             loss = None
#         else:
#             # B, T, C = logits.shape
#             # logits2 = logits.view(B * T, C)
#             # targets2 = targets.view(B * T)
#             loss = nn.functional.cross_entropy(logits, targets[:, -1])
#
#         return logits, loss
class MLP(NextToken):
    def __init__(self, vocab_size, block_size, n_embd, n_hidden, act=nn.Tanh):
        super().__init__()
        self.vocab_size = vocab_size
        H = n_hidden
        self.layers = nn.Sequential(
            nn.Embedding(vocab_size, n_embd),
            nn.Flatten(),
            # layer 1
            nn.Linear(n_embd * block_size, H),
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            act(),

            # layer 2
            nn.Linear(H, H),
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            act(),

            # layer 3
            nn.Linear(H, H),
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            act(),
            nn.Linear(H, vocab_size),
        )

    def forward(self, x, targets=None):
        logits = self.layers(x)  # (B,T,C)

        # return logits[:, -1, :], None

        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(logits, targets[:, -1])

        return logits, loss
