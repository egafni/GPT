import abc

import torch
from torch import nn


class NextToken(nn.Module):
    # def __init__(self, vocab_size):
    #     super().__init__()
    #     self.vocab_size = vocab_size

    @abc.abstractmethod
    def forward(self, idx, targets=None) -> (torch.Tensor, torch.Tensor):
        pass

    @torch.inference_mode
    def predict(self, X, max_tokens, prefix_inputs=False):
        # x is a batch
        y_hat = []
        for i in range(max_tokens):
            y_logit, loss = self(X)
            assert isinstance(y_logit, torch.Tensor)
            y_proba = torch.softmax(y_logit, axis=2)
            next_token = torch.multinomial(y_proba[:, -1, :], num_samples=1)
            X = torch.cat([X[:, 1:], next_token], axis=1)
            y_hat += [next_token]
        y_hat = torch.cat(y_hat, 1)
        if prefix_inputs:
            y_hat = torch.cat([X, y], axis=1)
        return y_hat


class BiGram(NextToken):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.token_embedding_table(x)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits2 = logits.view(B * T, C)
            targets2 = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits2, targets2)

        return logits, loss


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        # self._parameters = None

    def forward(self, x):
        return torch.permute(x, self.dims)


class MLP(NextToken):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        H = 128
        act = nn.Tanh
        self.seq = nn.Sequential(
            nn.Embedding(vocab_size, H),
            # layer 1
            nn.Linear(H, H),
            Permute((0, 2, 1)),  # B, T, C -> B, C, T
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            Permute((0, 2, 1)),  # B, C, T -> B, T, C
            act(),
            # layer 2
            nn.Linear(H, H),
            Permute((0, 2, 1)),  # B, T, C -> B, C, T
            nn.BatchNorm1d(H),  # so weird that this is B, C, T!!!
            Permute((0, 2, 1)),  # B, C, T -> B, T, C
            act(),

            nn.Linear(H, vocab_size),
        )

    def forward(self, x, targets=None):
        logits = self.seq(x)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits2 = logits.view(B * T, C)
            targets2 = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits2, targets2)

        return logits, loss
