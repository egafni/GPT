import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, fname, split, block_size):
        self.name = fname
        self.split = split
        self.block_size = block_size

        with open(fname) as fp:
            data = fp.read()

        vocab = sorted(set(data))
        self.vocab = vocab

        split_idx = int(len(data) * .9)
        if split == 'train':
            data = data[:split_idx]
        elif split == 'val':
            data = data[split_idx:]
        else:
            raise ValueError(f'invalid {split}')

        self.itos = {i: c for i, c in enumerate(vocab)}
        self.stoi = {c: i for i, c in enumerate(vocab)}

        data_i = torch.as_tensor([self.stoi[c] for c in data], dtype=torch.long)

        self.data = data
        self.data_i = data_i

    def decode(self, x: list[int] | torch.Tensor):
        if torch.is_tensor(x):
            x = x.tolist()
        return [self.itos[i] for i in x]

    def encode(self, x: list[str] | torch.Tensor):
        if torch.is_tensor(x):
            x = x.tolist()
        return [self.stoi[c] for c in x]

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        assert idx < len(self)
        return self.data_i[idx:idx + self.block_size], self.data_i[idx + 1:idx + self.block_size + 1]
