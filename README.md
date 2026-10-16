# OCQKUDSIN

`ocqkudsin` is a small-system PennyLane implementation of the OCQKUDSIN iteration.
It builds the normalized sine-filter operator, generates Lanczos states with
Chebyshev coefficients, measures the required moments, and projects the
Hamiltonian into the resulting subspace.
The solver is contained in `ocqkudsin.py`.

The implementation uses dense matrices and exact time evolution.

The [H4 collected-values PDF](docs/h4_collected_values_cqkud_landscape.pdf)
shows the first six states at epsilon 0.5, 0.8, and 1.2 in landscape form.
Its [full-precision data](docs/h4_collected_values_cqkud_landscape.json) can
be regenerated with `python3 examples/h4_collected_values.py`.
The [two-page Chebyshev coefficient explanation](docs/chebyshev_coefficient_growth_explained.pdf)
includes the growth plots, recurrence, and H4 values; its LaTeX source is in
the same directory.

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
0.5, 0.8, and 1.2. It writes `output.txt`, energy-convergence and
overlap condition-number plots, an absolute error plot against the
balanced-sector active-space FCI reference, and a Lanczos coefficient
plot (PNG and PDF) to `output/ocqkudsin/`. It also saves plots of each
Chebyshev coefficient's growth and a CSV of their full-precision values.
Use `--output-dir PATH` to choose another output directory.
The original `ocqkudsin` import name `QuantumOCQKUD` also remains available.
