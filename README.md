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

