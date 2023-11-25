import os

import torch
from torch.utils.data import DataLoader

from gpt.data import TextDataset
from gpt.models.mlp import MLP
from gpt.models.transformer import Transformer

REPO_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def fit(model, device, dl_train, dl_val, max_epochs, learning_rate, eval_freq=20):
    train_losses = []
    val_losses = {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    step = 0

    try:
        for epoch in range(1, max_epochs + 1):
            for i, (X, Y) in enumerate(iter(dl_train), start=1):
                step += 1
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
                    val_loss = eval(model, device, dl_val)
                    val_losses[step] = val_loss
                    lr = optimizer.param_groups[0]['lr']
                    print(f'{epoch}:{i}|s={step} loss: {loss.item():.4f} val_loss: {val_loss:.4f} lr: {lr}')

                train_losses.append(float(loss))

            print('epoch done')
    except KeyboardInterrupt:
        print('interrupted')

    return train_losses, val_losses


@torch.no_grad()
@torch.inference_mode()
def eval(model, device, dl_val):
    losses = []
    for X, Y in iter(dl_val):
        X = X.to(device)
        Y = Y.to(device)
        y_logit, loss = model(X, Y)
        losses.append(loss.item())
    return torch.mean(torch.tensor(losses)).item()


def get_experiments(block_size, vocab_size):
    return dict(
        mlp=MLP(
            vocab_size=vocab_size, block_size=block_size, n_embd=128, n_hidden=3
        ),
        transformer=Transformer(Transformer.Config(
            block_size=block_size,
            n_embd=384,
            n_heads=6,
            vocab_size=vocab_size,
            n_blocks=6,
            dropout=.2)
        )
    )


def run_experiments(max_epochs,
                    batch_size=64,
                    block_size=256,
                    learning_rate=3e-4):
    torch.manual_seed(1337)
    ds_train = TextDataset(f'{REPO_DIR}/input.txt', 'train', block_size=block_size)
    ds_val = TextDataset(f'{REPO_DIR}/input.txt', 'val', block_size=block_size)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    vocab_size = len(ds_train.vocab)

    for name, model in get_experiments(block_size, vocab_size).items():
        print(f'Running experiment: {name}')
        print('*' * 72)
        train_losses, val_losses = fit(model, dl_train, dl_val, max_epochs=max_epochs, learning_rate=learning_rate)
        print('*' * 72)


if __name__ == '__main__':
    run_experiments()
