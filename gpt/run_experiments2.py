import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from gpt.data import TextDataset
from gpt.models.mlp import MLP
from gpt.models.transformer import Transformer

REPO_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def batch_iter(dataset, batch_size, max_steps):
    """Epoch free data iterator"""
    i = 0
    while True:
        idxs = torch.randint(0, len(dataset), size=batch_size)
        yield torch.as_tensor([dataset[i] for i in idxs])
        i += 1
        if i >= max_steps:
            break


def fit(model, dl_train, dl_val, max_steps, learning_rate, eval_freq=50):
    device = model.device
    train_losses = []
    val_losses = {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step, (X, Y) in enumerate(iter(dl_train), start=1):
        # break
        X = X.to(device)
        Y = Y.to(device)
        y_logit, loss = model(X, Y)

        # break
        optimizer.zero_grad(True)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if step == 0 or step % eval_freq == 0:
            val_loss = eval(model, dl_val)
            val_losses[step] = val_loss
            lr = optimizer.param_groups[0]['lr']
            print(f'{step} loss: {loss.item()} val_loss: {val_loss} lr: {lr}')

        train_losses.append(float(loss))

    return train_losses, val_losses


@torch.no_grad()
@torch.inference_mode()
def eval(model, dl_val):
    device = model.device
    losses = []
    for X, Y in iter(dl_val):
        X = X.to(device)
        Y = Y.to(device)
        y_logit, loss = model(X, Y)
        losses.append(loss.item())
    return torch.mean(torch.tensor(losses)).item()


@dataclass
class Experiment:
    name: str
    model: torch.nn.Module
    block_size: int
    batch_size: int
    learning_rate: float


def get_all_experiments(block_size, vocab_size):
    yield Experiment(name='mlp',
                     model=MLP(vocab_size=vocab_size, block_size=block_size, n_embd=128, n_hidden=3),
                     batch_size=64,
                     learning_rate=3e-4,
                     block_size=block_size)

    yield Experiment(name='transformer',
                     model=Transformer(Transformer.Config(block_size=block_size,
                                                          n_embd=384,
                                                          n_heads=6,
                                                          vocab_size=vocab_size,
                                                          n_blocks=6,
                                                          dropout=.2)),
                     batch_size=64,
                     learning_rate=3e-4,
                     block_size=block_size
                     )


def run_experiments(max_steps):
    torch.manual_seed(1337)

    for experiment in get_all_experiments(block_size, vocab_size).items():
        ds_train = TextDataset(f'{REPO_DIR}/input.txt', 'train', block_size=block_size)
        ds_val = TextDataset(f'{REPO_DIR}/input.txt', 'val', block_size=block_size)

        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

        vocab_size = len(ds_train.vocab)

        print(f'Training: {experiment.name}')
        print('*' * 72)
        train_losses, val_losses = fit(experiment.model, dl_train, dl_val, max_steps=max_steps,
                                       learning_rate=experiment.learning_rate)
        print('*' * 72)


if __name__ == '__main__':
    run_experiments()
