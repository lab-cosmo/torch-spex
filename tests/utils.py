import torch

from collections import namedtuple

Graph = namedtuple("Graph", ("R_ij", "i", "j", "species"))


def to_graph(atoms, cutoff, dtype=torch.float32):
    from vesin import ase_neighbor_list

    i, j, D = ase_neighbor_list(
        "ijD", atoms, cutoff
    )  # they follow the R_ij = R_j - R_i convention
    species = atoms.get_atomic_numbers().astype(int)

    return Graph(
        torch.tensor(D, dtype=dtype),
        torch.tensor(i, dtype=torch.int),
        torch.tensor(j, dtype=torch.int),
        torch.tensor(species, dtype=torch.int),
    )


def combine_graphs(graphs):
    num_nodes = torch.tensor([g.species.shape[0] for g in graphs])
    offsets = torch.concatenate((torch.tensor([0]), torch.cumsum(num_nodes, dim=0)[:-1]))

    R_ij = torch.concatenate(tuple(g.R_ij for g in graphs), dim=0)
    species = torch.concatenate(tuple(g.species for g in graphs), dim=0)

    i = torch.concatenate(tuple(g.i + o for o, g in zip(offsets, graphs)), dim=0)
    j = torch.concatenate(tuple(g.j + o for o, g in zip(offsets, graphs)), dim=0)

    structures = torch.repeat_interleave(torch.arange(num_nodes.shape[0]), num_nodes)
    centers = torch.concatenate(tuple(torch.unique(g.i) for g in graphs))

    return Graph(R_ij, i, j, species), structures, centers
