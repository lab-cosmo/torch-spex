# `spex.engine`: Instantiation and de/serialisation

This sub-module deals with turning `torch.nn.Module`s into something that can be saved to and restored from disk. The essential problem to solve is that `torch` gives us good tools to store the parameters on disk, but the parameters need to be restored into an already-instantiated `Module`. So we need a way to re-initialise modules from some convenient and "saveable" description. For this, we use [`specable`](https://github.com/sirmarcel/specable)-style `dicts` that look like this:

```python
{"spex.cutoff.ShiftedCosine": {"cutoff": 5.0, "width": 1.0}}
```

Under the hood, all that happens (in `spec.py`) is that we split the "outer" key of the dict on the last `.`, and then import what comes after from what comes before. Then we put everything in the "inner dict" as `**inner` into the `__init__` method of the `nn.Module` that we just imported. So here, we basically run `from spex.cutoff import ShiftedCosine` and then `return ShiftedCosin(**inner)`.

This has the advantage of being trivially composable with custom code: All you need to do is to make it importable -- it can even be a file in the folder you're currently working in.

(As a sidenote, we don't use pickle because it does something similar, but in a much more opaque way -- this approach has the advantage of yielding human-readable model descriptions and being much easier to understand.)
