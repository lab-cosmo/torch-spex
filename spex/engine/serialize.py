import torch

from pathlib import Path

from .spec import from_dict, to_dict
from .yaml import read_yaml, write_yaml


def save(folder, module):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    spec = to_dict(module)
    params = module.state_dict()

    torch.save(params, folder / "params.torch")
    write_yaml(folder / "model.yaml", spec)


def load(folder):
    folder = Path(folder)

    spec = read_yaml(folder / "model.yaml")
    model = from_dict(spec)

    params = torch.load(folder / "params.torch", map_location="cpu", weights_only=True)

    model.load_state_dict(params)

    return model
