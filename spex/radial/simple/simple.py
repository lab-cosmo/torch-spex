import torch


class Simple(torch.nn.Module):
    """Base class for "simple" radial expansions.

    We take "simple" to mean that we define some family of function of fixed size,
    and simply expand some given distance ``r`` (up to ``cutoff``) in that basis.

    This base expansion into ``num_radial`` features can then optionally
    (``trainable``) be transformed with a linear layer into ``num_features``
    (defaulting to ``num_radial``).

    We can learn a separate transformation per degree (``per_degree=True``).

    Note:
        This class is not intended to be user-facing, it just separates the logic
        of computing the underlying expansion (via the ``expand`` method) from the
        learned transformations, and unifies interfacing with the rest of the package.

    """

    def __init__(
        self,
        cutoff,
        max_angular=3,
        num_radial=128,
        trainable=False,
        per_degree=False,
        num_features=None,
        **kwargs,
    ):
        super().__init__()

        self.spec = {
            "cutoff": cutoff,
            "max_angular": max_angular,
            "num_radial": num_radial,
            "trainable": trainable,
            "per_degree": per_degree,
            "num_features": num_features,
        }

        if not trainable:
            assert num_features is None, "trainable must be True to specify num_features"
            num_features = num_radial
            assert per_degree is False, "trainable must be True to specify per_degree=True"

        if trainable:
            if num_features is None:
                num_features = num_radial

        # needed for subclasses
        self.num_radial = num_radial

        # public interface
        self.cutoff = cutoff
        self.max_angular = max_angular
        self.n_per_l = [num_features for _ in range(self.max_angular + 1)]

        if trainable:
            if per_degree:
                self.transforms = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(self.num_radial, num_features)
                        for _ in range(self.max_angular + 1)
                    ]
                )
            else:
                transform = torch.nn.Linear(self.num_radial, num_features)
                self.transforms = torch.nn.ModuleList(
                    [transform for _ in range(self.max_angular + 1)]
                )

        else:
            # torchscript needs this to exist
            self.transforms = torch.nn.ModuleList(
                [NoOp() for _ in range(self.max_angular + 1)]
            )

    # -- to be implemented in sublass --

    def expand(self, r):
        """Compute the "raw" radial basis.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Expansion as a tensor of shape ``[pair, num_radial]``.
        """

        raise NotImplementedError

    # -- actual work --

    def forward(self, r):
        """Compute the radial basis.

        Args:
            r (Tensor): Input distances of shape ``[pair]``.

        Returns:
            Expansions as a per-degree list of tensors ``[pair, num_features]``.
        """

        expansion = self.expand(r)
        expansions = [transform(expansion) for transform in self.transforms]

        return expansions


class NoOp(torch.nn.Module):
    """Do nothing."""

    def forward(self, r):
        return r
