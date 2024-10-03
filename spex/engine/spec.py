import importlib
from copy import deepcopy

from .dicts import parse_dict


def to_dict(module):
    handle = f"{module.__module__}.{module.__class__.__name__}"

    if hasattr(module, "spec"):
        inner = deepcopy(module.spec)
    else:
        raise ValueError(
            f"{handle} does not have a spec attribute; cannot not be turned into a dict."
        )

    return {handle: inner}


def from_dict(dct, module=None):
    handle, inner = parse_dict(dct)

    inferred_module = ".".join(handle.split(".")[:-1])

    if inferred_module != "":
        # we prefer an explicitly given module to any default
        module = inferred_module
    else:
        assert module is not None, f"couldn't find module for {handle}"

    module = importlib.import_module(module)

    kind = handle.split(".")[-1]

    cls = getattr(module, kind)

    return cls(**inner)
