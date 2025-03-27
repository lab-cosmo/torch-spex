import numpy as np
import torch
from torch.nn import Module

MAX_SPLINE_POINTS = 10_000  # the same as in featomic


class DynamicSpliner(Module):
    """Spliner with dynamically chosen number of basis functions.

    For a function defined on ``[0, cutoff]`` (with many-dimensional output, for
    example different ``l`` and ``n`` channels), generate a splined version with a certain
    accuracy: We dynamically increase the number of splines until the mean absolute
    error (MAE) matches our target accuracy, evaluated in between the end points of
    the respective spline intervals.

    Cubic Hermite splines <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>
    are used. They are determined by ``spline_values`` and ``spline_derivatives``,
    i.e. the values and derivatives of the function to be splined, so both are needed.

    Inputs outside ``[0, cutoff]`` are clipped into that interval.

    Heavily inspired by the corresponding code in ``featomic``.
    """

    def __init__(
        self,
        cutoff,
        values_fn,
        derivatives_fn,
        accuracy=1e-8,
    ):
        """Initialise a DynamicSpliner.

        This performs the search for the correct number of splines, and then
        registers the resulting spline coefficients for later evaluation.

        Args:
            cutoff (float): We are working in the interval ``[0, cutoff]``.
            values_fn (Callable): Function mapping 1D input to multiple outputs
                ``[x1, x2, ...] -> [[f1(x1), f2(x2), ...], [f1(x2), ...]]``.
            derivatives_fn (Callable): Derivative of values_fn.
            accuracy (float): Accuracy until which we will add spline points.
        """
        super().__init__()

        start = 0.0
        n = 3  # divide [0, cutoff] into n-1 intervals with n boundaries
        MAE = float("inf")

        # Now we bisect each interpolation interval until we reach target accuracy,
        # i.e. new interval boundaries are added in the middle of previous ones
        # n=3: |    |    | -> n=5: | | | | | -> n=9: |||||||
        # everything is in double precision for now, it can be downcast later
        while MAE > accuracy:
            n = (n * 2) - 1
            spacing = (cutoff - start) / (n - 1)

            spline_positions = np.linspace(start, cutoff, n)

            spline_values = values_fn(spline_positions)
            spline_derivatives = derivatives_fn(spline_positions)
            spline_spacing = np.array(spacing)

            test_positions = np.linspace(
                start + spacing / 2, cutoff - spacing / 2, n - 1
            )  # in the middle of spline intervals

            actual_values = values_fn(test_positions)
            predicted_values = evaluate_splines(
                torch.tensor(test_positions),
                torch.tensor(spline_spacing),
                torch.tensor(spline_values),
                torch.tensor(spline_derivatives),
            ).numpy()

            MAE = np.mean(np.abs(actual_values - predicted_values))

            if n > MAX_SPLINE_POINTS:
                raise ValueError(
                    f"Reached maximum number of spline points ({MAX_SPLINE_POINTS})!"
                )

        spline_values = torch.tensor(spline_values, dtype=torch.get_default_dtype())
        spline_derivatives = torch.tensor(
            spline_derivatives, dtype=torch.get_default_dtype()
        )
        spline_spacing = torch.tensor(spline_spacing, dtype=torch.get_default_dtype())

        # we add one extra interval in case we encounter r=cutoff
        spline_values = torch.concat(
            (
                spline_values,
                torch.zeros_like(spline_values[0]).unsqueeze(0),
            ),
            dim=0,
        )
        spline_derivatives = torch.concat(
            (
                spline_derivatives,
                torch.zeros_like(spline_derivatives[0]).unsqueeze(0),
            ),
            dim=0,
        )

        self.cutoff = cutoff
        self.register_buffer("spline_values", spline_values)
        self.register_buffer("spline_derivatives", spline_derivatives)
        self.register_buffer("spline_spacing", spline_spacing)

    def forward(self, r):
        # r: [pair]
        r = torch.clip(r, min=0.0, max=self.cutoff)  # safety first

        return evaluate_splines(
            r, self.spline_spacing, self.spline_values, self.spline_derivatives
        )


def evaluate_splines(x, delta_x, spline_values, spline_derivatives):
    n = (torch.floor(x / delta_x)).to(dtype=torch.long)  # -> [pair]

    t = (x - n * delta_x) / delta_x  # -> [pair]
    t_2 = t**2
    t_3 = t**3

    h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0  # -> [pair]
    h10 = t_3 - 2.0 * t_2 + t
    h01 = -2.0 * t_3 + 3.0 * t_2
    h11 = t_3 - t_2

    h00 = h00.unsqueeze(-1)
    h10 = h10.unsqueeze(-1)
    h01 = h01.unsqueeze(-1)
    h11 = h11.unsqueeze(-1)

    p_k = torch.index_select(spline_values, dim=0, index=n)  # -> [pair, dim]
    p_k_1 = torch.index_select(spline_values, dim=0, index=n + 1)

    m_k = torch.index_select(spline_derivatives, dim=0, index=n)
    m_k_1 = torch.index_select(spline_derivatives, dim=0, index=n + 1)

    return (
        h00 * p_k + h10 * delta_x * m_k + h01 * p_k_1 + h11 * delta_x * m_k_1
    )  # -> [pair, dim]
