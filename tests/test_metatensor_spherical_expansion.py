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

        exp = SphericalExpansion()
        exp = torch.jit.script(exp)

        exp(R_ij, i, j, species, structures, centers)

    def test_different_backends(self):
        from spex.metatensor.spherical_expansion import SphericalExpansion

        for device in ("cpu", "cuda", "mps"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue
            if device == "mps":
                # sphericart does not support MPS; skip
                continue

            exp = SphericalExpansion()

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

    def test_molecules_vs_rascaline(self):
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

        rascaline_calc = get_rascaline_calculator(cutoff, max_angular, max_radial, species)
        spex_calc = get_spex_calculator(cutoff, max_angular, max_radial, species)

        reference = rascaline_calc.compute(systems)
        ours = spex_calc(graph.R_ij, graph.i, graph.j, graph.species, structures, centers)

        compare(reference, ours, atol=1e-5)

    def test_bulk_vs_rascaline(self):
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

        rascaline_calc = get_rascaline_calculator(cutoff, max_angular, max_radial, species)
        spex_calc = get_spex_calculator(cutoff, max_angular, max_radial, species)

        reference = rascaline_calc.compute(systems)
        ours = spex_calc(graph.R_ij, graph.i, graph.j, graph.species, structures, centers)

        compare(reference, ours, atol=1e-5)


def compare(first, second, atol=1e-11):
    # compare first, a TensorMap emitted by rascaline (which is using metatensor) with
    #         second, a TensorMap emitted by our code (which uses metatensor.torch)

    import random

    import metatensor as standard_mt
    import metatensor.torch as mt

    # we need to save/read back the TensorMap to switch from vanilla metatensor to
    # the metatensor.torch interface... we do it via a temporary file

    file = f"tmp_{random.randint(0, 100)}.npz"

    standard_mt.save(file, first)
    first = mt.load(file)

    # delete *before* the test could fail
    pathlib.Path(file).unlink()

    assert mt.allclose(first, second, atol=atol)


def get_spex_calculator(cutoff, max_angular, max_radial, species, spliner_accuracy=1e-8):
    from spex.metatensor.spherical_expansion import SphericalExpansion

    return SphericalExpansion(
        radial={
            "LaplacianEigenstates": {
                "cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "trim": False,
                "spliner_accuracy": spliner_accuracy,
            }
        },
        angular="SphericalHarmonics",
        species={"Orthogonal": {"species": species}},
        cutoff_function={"ShiftedCosine": {"width": 0.5}},
    )


def get_rascaline_calculator(
    cutoff, max_angular, max_radial, species, spliner_accuracy=1e-8
):
    import rascaline

    max_radial += 1

    spliner = rascaline.utils.SoapSpliner(
        cutoff=cutoff,
        max_radial=max_radial,
        max_angular=max_angular,
        basis=rascaline.utils.SphericalBesselBasis(
            cutoff=cutoff, max_radial=max_radial, max_angular=max_angular
        ),
        density=rascaline.utils.DeltaDensity(),
        accuracy=spliner_accuracy,
    )

    splined_basis = spliner.compute()

    return rascaline.SphericalExpansion(
        cutoff=cutoff,
        max_radial=max_radial,
        max_angular=max_angular,
        center_atom_weight=0.0,
        radial_basis=splined_basis,
        atomic_gaussian_width=-1.0,  # will not be used due to the delta density above
        cutoff_function={"ShiftedCosine": {"width": 0.5}},
    )
