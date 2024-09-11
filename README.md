# `torch-spex`: spherical expansions of atomic neighbourhoods

## Installation

`spex` requires [`torch`](https://pytorch.org/get-started/locally/) and [`sphericart`](https://sphericart.readthedocs.io/en/latest/installation.html) (with `torch` support) to be installed manually.

Using the [`metatensor`](https://docs.metatensor.org/latest/installation.html) interface requires the installation of the `torch` version.

Running the tests additionally requires [`rascaline`](https://luthaf.fr/rascaline/latest/get-started/installation.html) (with `torch`).

```bash
# install the appropriate version of torch for your setup
pip install "sphericart[torch] @ git+https://github.com/lab-cosmo/sphericart"

pip install metatensor[torch]
# make sure that the rust compiler is available
pip install git+https://github.com/luthaf/rascaline#subdirectory=python/rascaline-torch
```

Once these depencies are present, you should be able to install `spex` as usual:

```
git clone git@github.com:sirmarcel/spex-dev.git
cd spex-dev
pip install -e .

# or (to install pytest, etc)
pip install -e .[dev]

```

## Development

`spex` uses `ruff` for formatting. Please use the [pre-commit hook](https://pre-commit.com) to make sure that any contributions are formatted correctly, or run `ruff format . && ruff check --fix .`.