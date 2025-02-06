# `torch-spex`: spherical expansions of atomic neighbourhoods

`torch-spex` computes spherical expansions of atomic neighbourhoods in `torch`. It provides both a ready-to-use calculator, `spex.SphericalExpansion`, as well as building blocks required to implement custom expansions. It's fully compatible with TorchScript and also has a `metatensor` calculator. As of now, outputs are precisely equivalent to `rascaline` for matching settings, while typically outperforming it by a significant margin on GPUs.

A spherical expansion is commonly used as the input for machine learning models of molecules and materials, but can, in principle, be used for other tasks related to learning of labelled point clouds. In the language of atomistic ML, a spherical expansion computes a fixed-size representation of the arrangement of atoms $j$ in a neighbourhood around a central atom $i$, based on the vectors connecting $i$ and $j$, $\vec R_{ij}$, and the chemical species labels $Z_j$ (typically the atomic number). It works by expanding the different ingredients as follows:

```math
A^{lm}_{nc} = \sum_j C_c(Z_j) \left( R^l_n(|\vec R_{ij}|) S^{lm}(\vec R_{ij}) \right) f_{\text{cut}}(|\vec R_{ij}|)
```

where $C_c$ is a function that maps $Z_j$ to some embedding space with index $c$, $R$ is a function that embeds the distance between $i$ and $j$, and $S_{lm}$ is a spherical harmonic, solid harmonic, or other irreducible representation of the rotation group. $f_{\text{cut}}$ is a cutoff function that ensures that pairwise contributions decay to zero at the cutoff radius. The resulting object has four indices, and since there are different numbers of $m$ per $l$, it is returned as a `list` of `torch.Tensor`, with every tensor arranged as $[m, n, c]$ and $l$ indexing the (outer) list. Since we compute everything for whole structures at once, $i$ is an additional, leading, index to the inner tensors. By construction, this object is *equivariant* under rotations and can be further processed in equivariant neural networks, or turned into invariant features for regression.

***

From a technical standpoint, we decompose the problem exactly as discussed above: `spex.species` provides options for `C`, `spex.radial` for `R`, and `spex.angular` for `S`. Finally, `spex.cutoff` provides options for $f_{\text{cut}}$. `spex.SphericalExpansion` multiplies it all together and performs the sum over $j$. We currently support:

Species embeddings (`spex.species`):

- `Alchemical`: Typical embedding of fixed size, can be understood as mapping elements to linear combinations of "pseudo species",
- `Orthogonal`: One-hot embedding, keeping each species in a separate subspace.


Radial embeddings (`spex.radial`):

- `LaplacianEigenstates` and `PhysicalBasis`: Physics-inspired basis functions (splined for efficiency),
- `Bernstein`: Bernstein polynomials.

Angular embeddings (`spex.angular`):

- `SphericalHarmonics`: (real) spherical harmonics (provided by [`sphericart`](https://github.com/lab-cosmo/sphericart/)),
- `SolidHarmonics`: solid harmonics (i.e., spherical harmonics multiplied by `r**l`).


Cutoff functions (`spex.cutoff`):

- `ShiftedCosine`: Cosine shifted such that it goes from 1 to 0 within a certain `width` of the cutoff radius
- `Step`: Step function, or *hard* cutoff. Not advisable to use in practice at it makes the potential-energy surface not continuously differentiable and the resulting force field non-conservative.

The interfaces to each of these types of components is defined in the readmes of the sub-packages. For custom components, please make sure to implement the corresponding interface.

## Installation

You should be able to install `spex` as usual:

```
git clone git@github.com:sirmarcel/spex-dev.git
cd spex-dev
pip install -e .

# or (to install pytest, etc)
pip install -e ".[dev]"

# or (for the physical basis)
pip install -e ".[physical]"
```

You may have to manually install `torch` for your particular setup before doing this.

## Usage

Regrettably, there is currently no nicely rendered documentation for this package. Instead, you are expected and encouraged to look into the code itself, where you will find docstrings on all public-facing functions, as well as docsctrings at the sub-package level (in the `__init__.py`) that explain in more detail what is going on.

### Instantiating a `SphericalExpansion`

`spex` components can all be instantiated from `dict`s that take the form `{ClassName: {"arg": 1, ...}}`, similar to [`featomic` (fka `rascaline`)](https://github.com/metatensor/featomic) and inspired by [`specable`](https://github.com/sirmarcel/specable). This feature is used heavily in `SphericalExpansion`, which accepts this style of `dict` to specify the different embeddings to use. If a `.` is present in `ClassName`, for example `mything.SpecialRadial`, `spex` will try to import `SpecialRadial` from `mything`, so we have a basic plug-in system included for free! In all cases, the "inner" `dict` will simple be `**splatted` into the `__init__` of `ClassName`.

Here is a full example for a `SphericalExpansion`:

```python

from spex import SphericalExpansion

exp = SphericalExpansion(
    5.0,  # cutoff radius
    max_angular=3,  # l = 0,1,2,3
    radial={"LaplacianEigenstates": {"max_radial": 8}},
    angular="SphericalHarmonics",
    species={"Alchemical": {"pseudo_species": 4}},
    cutoff_function={"ShiftedCosine": {"width": 0.5}},
)

```

Equivalently, we can write this out in `.yaml`:

```yaml
spex.SphericalExpansion:
  cutoff: 5.0
  max_angular: 3
  radial:
    LaplacianEigenstates:
      max_radial: 8
  angular: SphericalHarmonics
  species:
    Alchemical:
      pseudo_species: 4
  cutoff_function:
    ShiftedCosine:
      width: 0.5

```

From this `.yaml` file, we can instantiate with

```python
from spex import from_dict, read_yaml

exp = from_dict(read_yaml("spex.yaml"))
```

### Saving and Loading (bonus feature)

This is already one half of everything we need for a general tool to save and load `torch` models, since `torch` gives us the ability to save the weights (and other parameters) of `torch.nn.Module` with `module.state_dict()`, but it doesn't store how to instantiate a template to load the weights into. The lightweight `dict`-based approach here manages that. For convenience, `spex` puts the two things together: `spex.save` will make a folder with `params.torch` for weights and `model.yaml` for the `dict` (we call it a `spec`), and `spex.load` will instantiate the model and load the weights. This is not needed for most of the components from `spex`, since currently only some radial basis functions have learnable parameters, but it can also be used to improvise checkpointing for experiments.

### Customising `SphericalExpansion`

There are two way to customise the expansion: (a) with custom embeddings, or (b) in other ways. (a) is supported by the "plugin" system: You can write a class that conforms to the interfaces defined in the respective sub-packages of `spex` (`spex.radial`, `spex.angular`, `spex.species`), and then just pass, for example, `radial={"mypackage.MyClass": {"my_arg" .."}}` to the `SphericalExpansion`. (Note that `mypackage` can also be a `mypackage.py` in the same folder as your current script.) All other customisations, (b), are *not* supported intrinsically by `spex` and you are expected to copy `spherical_expansion.py` and hack away. Do not hesitate to ask if you have trouble with any particular plan, we're happy to help.

### What about SOAP?

`spex` only computes a spherical expansion and not any downstream descriptors. The well-known SOAP descriptor can be obtained, for example, via:

```python
expansion = exp(R_ij, i, j, species)  # -> [[i, m, n, c], ...]
soap = [torch.einsum("imnc,imNC->inNcC", e, e) for e in expansion]  # -> [[i, n1, n2, c1, c2], [...], ...]
```

Note that this may produce very large features. You may want to consider other, "contracted", approaches where inner instead of outer products are performed. How to do this is beyond the scope of this readme. :)

## Development

`spex` uses `ruff` for formatting. Please use the [pre-commit hook](https://pre-commit.com) to make sure that any contributions are formatted correctly, or run `ruff format . && ruff check --fix .`.

We generally adhere to the [Google style guidelines](https://google.github.io/styleguide/pyguide.html). Note in particular how docstrings are formatted, and the docstrings are only encouraged for public-facing API, unless required to explain something. All docstrings and particularly comments are expected to be kept to a minimum and to be concise. Make sure you edit your LLM output accordingly.

Please review the development readme in `spex/README.md` for further information.
