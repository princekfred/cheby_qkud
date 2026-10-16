# CQKUD

`cqkud` is a small-system PennyLane implementation of the OCQKUD iteration.
It builds the normalized sine-filter operator, generates Lanczos states with
Chebyshev coefficients, measures the required moments, and projects the
Hamiltonian into the resulting subspace.

`ocqkudsin` provides an explicit OCQKUDSIN import using the same sine-filter
solver. Both names use the numerical stability improvements in `cqkud.py`.

The implementation uses dense matrices and exact time evolution.

The [two-page mathematical implementation](docs/cqkud_mathematical_implementation.pdf)
explains the sine-filter solver, projected Ritz energy, and collected state
expectation with its global factor absorbed into each coefficient. Its
editable LaTeX source is in the same directory.
The [H4 collected-values PDF](docs/h4_collected_values_cqkud_landscape.pdf)
shows the first six states at epsilon 0.5, 0.8, and 1.2 in landscape form.
Its [full-precision data](docs/h4_collected_values_cqkud_landscape.json) can
be regenerated with `python3 examples/h4_collected_values.py`.
The [OCQKUDSIN mathematical implementation](docs/ocqkudsin_math_implementation.pdf)
describes the shared sine-filter solver.

## Install

Use Python 3.9 or newer.

```bash
python -m pip install -e .
```

## Run

From the repository root:

```bash
python -m examples.h4
```

The H4 example uses the bundled Hamiltonian and Hartree-Fock reference from
the `ocqkudsin` square-H4 comparison. This keeps its Krylov starting state
and convergence reproducible across PennyLane versions. To build the
Hamiltonian with the installed PennyLane version instead, run
`python -m examples.h4 --rebuild`; orbital choices may then change which
eigenstates the Hartree-Fock reference can reach.

The single H4 input in `examples/h4.py` runs only OCQKUDSIN at epsilon
0.5, 0.8, and 1.2. It writes `output.txt`, an energy-convergence plot,
an overlap condition-number plot, and an absolute error plot against the
balanced-sector active-space FCI reference to `output/ocqkudsin/`.
Use `--output-dir PATH` to choose another output directory.
The original `ocqkudsin` import name `QuantumOCQKUD` also remains available.
