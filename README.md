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
python3 examples/h4_lin.py
```

The linear H4 input builds its Hamiltonian directly from four hydrogen atoms
spaced 3.0 Angstrom apart, using STO-3G and four active electrons in four
active orbitals. It runs only OCQKUDSIN at epsilon 0.5, 0.8, and 1.2, then
writes the energies, overlap condition numbers, coefficient plots, and
Chebyshev coefficient CSV to `output/h4_lin/`. No archived Hamiltonian is used.

For the square H4 example, run `python3 examples/h4_sq.py`. It builds the
1.5 Angstrom square Hamiltonian directly with PennyLane, reports FCI errors
in Ha, and writes results to `output/h4_sq/`. Use `--output-dir PATH` with
either input to choose another output directory. The collected-values PDF
above retains its separate archived Hamiltonian for reproducibility.
The original `ocqkudsin` import name `QuantumOCQKUD` also remains available.
