# todo: this test is not really checking against featomic
#       -- will remove magic saved stuff later
import torch

import pathlib
from unittest import TestCase


class TestMetatenorSphericalExpansion(TestCase):
    def test_jit(self):
        from spex.metatensor.spherical_expansion import SphericalExpansion

        R_ij = torch.randn((6, 3))
        i = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
        j = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.int64)
        species = torch.tensor([8, 8, 64])

        centers = torch.tensor([0, 1, 2])
        structures = torch.tensor([0, 0, 0])

        exp = SphericalExpansion(5.0)
        exp = torch.jit.script(exp)

        exp(R_ij, i, j, species, structures, centers)

    def test_different_backends(self):
        from spex.metatensor.spherical_expansion import SphericalExpansion

        for device in ("cpu", "cuda"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue

            exp = SphericalExpansion(5.0)

            R_ij = torch.randn((6, 3))
            i = torch.tensor([0, 0, 1, 1, 2, 2])
            j = torch.tensor([1, 2, 0, 2, 0, 1])
            species = torch.tensor([8, 8, 64])

            centers = torch.tensor([0, 1, 2])
            structures = torch.tensor([0, 0, 0])

            R_ij = R_ij.to(device)
            i = i.to(device)
            j = j.to(device)
            species = species.to(device)
            exp = exp.to(device)

            centers = centers.to(device)
            structures = structures.to(device)

            exp(R_ij, i, j, species, centers, structures)

    def test_molecules_vs_featomic(self):
        from ase.build import molecule
        from utils import combine_graphs, to_graph

        cutoff = 1.7
        max_radial = 10
        max_angular = 10
        species = [1, 8]

        atoms = molecule("H2O")
        systems = []
        for i in range(10):
            atoms.rattle(seed=i)
            systems.append(atoms.copy())

        graph, structures, centers = combine_graphs(
            [to_graph(atoms, cutoff) for atoms in systems]
        )

        featomic_calc = get_featomic_calculator(cutoff, max_angular, max_radial, species)
        spex_calc = get_spex_calculator(cutoff, max_angular, max_radial, species)

        reference = featomic_calc.compute(systems)
        ours = spex_calc(graph.R_ij, graph.i, graph.j, graph.species, structures, centers)

        compare(reference, ours, atol=1e-5, name="mol")

    def test_bulk_vs_featomic(self):
        from ase.build import bulk
        from utils import combine_graphs, to_graph

        cutoff = 5.0
        max_radial = 6
        max_angular = 4
        species = [1, 2]

        atoms = bulk("Ar", cubic=True)
        atoms.set_atomic_numbers([1, 2, 1, 2])
        atoms = atoms * [3, 3, 3]
        systems = []
        for i in range(10):
            atoms.rattle(seed=i)
            systems.append(atoms.copy())

        graph, structures, centers = combine_graphs(
            [to_graph(atoms, cutoff) for atoms in systems]
        )

        featomic_calc = get_featomic_calculator(cutoff, max_angular, max_radial, species)
        spex_calc = get_spex_calculator(cutoff, max_angular, max_radial, species)

        reference = featomic_calc.compute(systems)
        ours = spex_calc(graph.R_ij, graph.i, graph.j, graph.species, structures, centers)

        compare(reference, ours, atol=1e-5, name="bulk")


def compare(first, second, atol=1e-11, name=None):
    import random

    import metatensor as standard_mt
    import metatensor.torch as mt

    if name is None:
        file = pathlib.Path(__file__).parent / f"tmp_{random.randint(0, 100)}.npz"
        standard_mt.save(str(file), first)
        first = mt.load(str(file))
        file.unlink()
    else:
        file = pathlib.Path(__file__).parent / f"tmp_{name}.npz"
        first = mt.load(str(file))

    assert mt.allclose(first, second, atol=atol)


def get_spex_calculator(cutoff, max_angular, max_radial, species, spliner_accuracy=1e-8):
    from spex.metatensor.spherical_expansion import SphericalExpansion

    calc = SphericalExpansion(
        cutoff=cutoff,
        max_angular=max_angular,
        radial={
            "LaplacianEigenstates": {
                "max_radial": max_radial,
                "trim": False,
                "spliner_accuracy": spliner_accuracy,
            }
        },
        angular="SphericalHarmonics",
        species={"Orthogonal": {"species": species}},
        cutoff_function={"ShiftedCosine": {"width": 0.5}},
    )
    calc = calc.to(dtype=torch.float64)

    return calc


def get_featomic_calculator(
    cutoff, max_angular, max_radial, species, spliner_accuracy=1e-8
):
    import featomic
    from featomic.splines import SoapSpliner

    spliner = SoapSpliner(
        cutoff=featomic.cutoff.Cutoff(
            radius=cutoff, smoothing=featomic.cutoff.ShiftedCosine(width=0.5)
        ),
        density=featomic.density.DiracDelta(),
        basis=featomic.basis.LaplacianEigenstate(
            radius=cutoff,
            max_radial=max_radial,
            max_angular=max_angular,
            spline_accuracy=spliner_accuracy,
        ),
    )

    hypers = spliner.get_hypers()
    return featomic.SphericalExpansion(**hypers)
