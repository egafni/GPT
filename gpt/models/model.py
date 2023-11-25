import abc

import torch
from torch import nn


class NextToken(nn.Module):
    @abc.abstractmethod
    def forward(self, idx, targets=None) -> (torch.Tensor, torch.Tensor):
        pass

    @torch.inference_mode()
    def predict(self, X, max_tokens, prefix_inputs=False):
        # x is a batch
        y_hat = []
        for i in range(max_tokens):
            y_logit, loss = self(X)
            assert isinstance(y_logit, torch.Tensor)
            if y_logit.ndim == 2:
                y_proba = torch.softmax(y_logit, axis=1)
                next_token = torch.multinomial(y_proba, num_samples=1)
            elif y_logit.ndim == 3:
                y_proba = torch.softmax(y_logit, axis=2)
                next_token = torch.multinomial(y_proba[:, -1, :], num_samples=1)

            X = torch.cat([X[:, 1:], next_token], axis=1)
            y_hat += [next_token]
        y_hat = torch.cat(y_hat, 1)
        if prefix_inputs:
            y_hat = torch.cat([X, y_hat], axis=1)
        return y_hat
