# CQKUD

`cqkud` is a small-system PennyLane implementation of the OCQKUD iteration.
It builds the normalized sine-filter operator, generates Lanczos states with
Chebyshev coefficients, measures the required moments, and projects the
Hamiltonian into the resulting subspace.

The implementation uses dense matrices, exact time evolution, and generic
state preparation. It is intended for small systems and demonstrations.

## Install

```bash
python -m pip install -e .
```

## Example

```python
import numpy as np

from ocqkud import QuantumOCQKUD, build_molecular_problem

problem = build_molecular_problem(
    symbols=["H", "H"],
    geometry_angstrom=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    active_electrons=2,
    active_orbitals=2,
)

result = QuantumOCQKUD(
    problem,
    epsilon=0.5,
    max_dimension=6,
).run()

print("Lanczos alpha:", result.alpha)
print("Lanczos beta:", result.beta)
print("Projected energies (Hartree):", result.physical_energies)
```

Set `shots=None` for analytic simulator measurements, or pass a positive
integer for finite-shot measurements. `epsilon=0` uses the direct Hamiltonian
moment circuits.

`build_molecular_problem` uses PennyLane's Jordan-Wigner mapping and
Hartree-Fock reference state. For a custom Hamiltonian, construct a
`MolecularProblem` directly.
