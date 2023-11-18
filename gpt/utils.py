import pandas as pd
import torch


def model_info(model, x):
    ds = [dict(name='X', out_size=tuple(x.shape))]
    for layer_name, layer in model.named_children():
        x = layer(x)
        nparams = sum(x.nelement() for x in layer.parameters())
        params = {n: tuple(p.shape) for n, p in layer.named_parameters()}
        d = dict(name=layer_name, params=params, out_size=tuple(x.shape), nparams=nparams)
        ds.append(d)

    return pd.DataFrame(ds)


def shape(x, shape):
    if x.shape != shape:
        raise ValueError(f'Expected shape {shape}, got {x.shape}')

def nan_hook(model):
    def nan_hook(self, inp, output):
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output

        for i, out in enumerate(outputs):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                                   out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    for submodule in model.modules():
        submodule.register_forward_hook(nan_hook)