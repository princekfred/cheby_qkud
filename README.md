# CQKUD

`cqkud` is a small-system PennyLane implementation of the OCQKUD iteration.
It builds the normalized sine-filter operator, generates Lanczos states with
Chebyshev coefficients, measures the required moments, and projects the
Hamiltonian into the resulting subspace.

The implementation uses dense matrices and exact time evolution.

## Install

Use Python 3.10 or newer.

```bash
python -m pip install -e .
```

## Run

From the repository root:

```bash
python -m unittest discover -s tests
python -m examples.h4
```

The H4 example writes the energy estimate at each Krylov dimension for two
epsilon values and their overlap-matrix condition numbers to
`output/output.txt`, and saves a condition-number plot as
`output/condition_number.png` plus an energy-convergence plot as
`output/energy_convergence.png`. It also computes an exact active-space FCI
reference for the balanced 2-alpha/2-beta sector and saves the absolute energy
error plot as `output/energy_error_fci.png`. The output directory is created
automatically. Results can vary with the PennyLane version used to generate
the molecular Hamiltonian.
