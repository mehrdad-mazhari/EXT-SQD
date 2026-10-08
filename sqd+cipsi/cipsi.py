"""Epstein--Nesbet CIPSI importance scoring.

The routines in this module are independent of a particular determinant
representation.  A caller supplies the Hamiltonian couplings between external
determinants and the current variational space.  This makes the scoring usable
with PySCF, Qiskit SQD, or a custom Slater--Condon implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CIPSIScores:
    """CIPSI data for a collection of external determinants."""

    scores: NDArray[np.float64]
    pt2_contributions: NDArray[np.float64]
    first_order_amplitudes: NDArray[np.complex128]
    residuals: NDArray[np.complex128]
    denominators: NDArray[np.float64]

    @property
    def pt2_correction(self) -> float:
        """Return the total Epstein--Nesbet second-order correction."""

        return float(np.sum(self.pt2_contributions))


def _regularize_denominators(
    denominators: NDArray[np.float64], level_shift: float
) -> NDArray[np.float64]:
    """Keep small denominators finite without changing their sign."""

    if level_shift <= 0:
        raise ValueError("level_shift must be positive")
    signs = np.where(denominators < 0.0, -1.0, 1.0)
    return signs * np.maximum(np.abs(denominators), level_shift)


def cipsi_scores(
    energy: float,
    coefficients: ArrayLike,
    external_diagonal: ArrayLike,
    couplings: ArrayLike,
    *,
    level_shift: float = 1e-8,
) -> CIPSIScores:
    """Score external determinants with Epstein--Nesbet CIPSI.

    For external determinant ``a`` and variational determinants ``i``, this
    computes ``r_a = sum_i H_ai c_i`` and

    ``delta_E_a = |r_a|**2 / (energy - H_aa)``.

    Determinants are ranked by ``abs(delta_E_a)``.  The signed contributions
    are retained so their sum can be used as a PT2 convergence diagnostic.

    Args:
        energy: Current variational electronic energy.
        coefficients: Normalized CI coefficients, shape ``(n_variational,)``.
        external_diagonal: Values ``H_aa``, shape ``(n_external,)``.
        couplings: Matrix ``H_ai``, shape
            ``(n_external, n_variational)``.
        level_shift: Minimum absolute Epstein--Nesbet denominator.  This guards
            against intruder states and division by zero.
    """

    coeff = np.asarray(coefficients)
    diagonal = np.asarray(external_diagonal, dtype=float)
    h_ai = np.asarray(couplings)

    if coeff.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    if diagonal.ndim != 1:
        raise ValueError("external_diagonal must be one-dimensional")
    if h_ai.ndim != 2:
        raise ValueError("couplings must be a two-dimensional matrix")
    expected_shape = (diagonal.size, coeff.size)
    if h_ai.shape != expected_shape:
        raise ValueError(
            f"couplings has shape {h_ai.shape}; expected {expected_shape}"
        )
    if not np.isfinite(energy):
        raise ValueError("energy must be finite")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("coefficients must be finite")
    if not np.all(np.isfinite(diagonal)) or not np.all(np.isfinite(h_ai)):
        raise ValueError("Hamiltonian elements must be finite")

    residuals = h_ai @ coeff
    return cipsi_scores_from_residuals(
        energy,
        diagonal,
        residuals,
        level_shift=level_shift,
    )


def cipsi_scores_from_residuals(
    energy: float,
    external_diagonal: ArrayLike,
    residuals: ArrayLike,
    *,
    level_shift: float = 1e-8,
) -> CIPSIScores:
    """Score determinants when ``r_a = <D_a|H|Psi>`` is precomputed."""

    diagonal = np.asarray(external_diagonal, dtype=float)
    residual = np.asarray(residuals)
    if diagonal.ndim != 1 or residual.ndim != 1:
        raise ValueError("external_diagonal and residuals must be one-dimensional")
    if diagonal.shape != residual.shape:
        raise ValueError("external_diagonal and residuals must have the same shape")
    if not np.isfinite(energy):
        raise ValueError("energy must be finite")
    if not np.all(np.isfinite(diagonal)) or not np.all(np.isfinite(residual)):
        raise ValueError("Hamiltonian elements and residuals must be finite")

    denominators = _regularize_denominators(energy - diagonal, level_shift)
    first_order = residual / denominators
    pt2 = np.abs(residual) ** 2 / denominators

    return CIPSIScores(
        scores=np.asarray(np.abs(pt2), dtype=float),
        pt2_contributions=np.asarray(pt2, dtype=float),
        first_order_amplitudes=np.asarray(first_order, dtype=complex),
        residuals=np.asarray(residual, dtype=complex),
        denominators=denominators,
    )


def select_top_cipsi(
    scores: ArrayLike,
    max_determinants: int,
    *,
    score_threshold: float = 0.0,
) -> NDArray[np.int64]:
    """Return indices of the strongest determinants in descending order.

    Sorting is stable, so equal scores retain their input ordering and runs are
    reproducible.
    """

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if max_determinants < 0:
        raise ValueError("max_determinants must be non-negative")
    if score_threshold < 0:
        raise ValueError("score_threshold must be non-negative")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")

    eligible = np.flatnonzero(values >= score_threshold)
    order = np.argsort(-values[eligible], kind="stable")
    return np.asarray(eligible[order[:max_determinants]], dtype=np.int64)
