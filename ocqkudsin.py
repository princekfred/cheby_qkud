"""Public OCQKUDSIN interface for the sine-filter solver.

The OCQKUDSIN working operator is ``sin(epsilon * H / pauli_bound)``. Its
Lanczos recurrence, circuit moments, Chebyshev state construction, and Ritz
projection are implemented in :mod:`cqkud`. Keeping one solver avoids the
two interfaces developing different numerical behavior.
"""

from cqkud import (
    MolecularProblem,
    OCQKUDResult,
    QuantumOCQKUD as _QuantumOCQKUD,
    balanced_spin_sector,
    build_molecular_problem,
    exact_sector_energy,
)


class QuantumOCQKUDSIN(_QuantumOCQKUD):
    """OCQKUDSIN with ``X_epsilon = sin(epsilon * H / pauli_bound)``."""


# Preserve the class name used by the original ocqkudsin module.
QuantumOCQKUD = QuantumOCQKUDSIN
OCQKUDSINResult = OCQKUDResult


__all__ = [
    "MolecularProblem",
    "OCQKUDResult",
    "OCQKUDSINResult",
    "QuantumOCQKUD",
    "QuantumOCQKUDSIN",
    "balanced_spin_sector",
    "build_molecular_problem",
    "exact_sector_energy",
]
