"""Run square H4 with OCQKUD at epsilon=0.8 and 1.2."""

from __future__ import annotations

from contextlib import redirect_stdout
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cqkud import QuantumOCQKUD, build_molecular_problem, exact_sector_energy


GEOMETRY_ANGSTROM = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5],
        [0.0, 1.5, 0.0],
        [0.0, 1.5, 1.5],
    ]
)
EPSILONS = (0.8, 1.2)
MAX_DIMENSION = 8
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_FILE = OUTPUT_DIR / "output.txt"
CONDITION_PLOT_FILE = OUTPUT_DIR / "condition_number.png"
ENERGY_PLOT_FILE = OUTPUT_DIR / "energy_convergence.png"
FCI_ERROR_PLOT_FILE = OUTPUT_DIR / "energy_error_fci.png"


def main() -> None:
    problem = build_molecular_problem(
        symbols=["H", "H", "H", "H"],
        geometry_angstrom=GEOMETRY_ANGSTROM,
        active_electrons=4,
        active_orbitals=4,
        basis="sto-3g",
    )

    fci_energy = exact_sector_energy(problem, n_alpha=2, n_beta=2)
    print("Square H4, side length 1.5 Angstrom, STO-3G")
    print(f"Pauli coefficient bound: {problem.pauli_bound:.12f} Hartree")
    print(f"FCI energy (active space, 2 alpha + 2 beta): {fci_energy:.12f} Hartree")
    condition_data = []
    energy_data = []
    fci_error_data = []
    for epsilon in EPSILONS:
        result = QuantumOCQKUD(
            problem,
            epsilon=epsilon,
            max_dimension=MAX_DIMENSION,
        ).run()

        print(f"\nepsilon = {epsilon:g}")
        print(
            " K   Energy (Hartree)      FCI error (mHa)       alpha"
            "          beta      overlap condition"
        )
        for row in result.history:
            k = row["K"]
            alpha = result.alpha[k - 1]
            beta_next = row["beta_next"]
            fci_error_millihartree = abs(row["energy_hartree"] - fci_energy) * 1000
            print(
                f"{k:2d}  {row['energy_hartree']:16.12f}  "
                f"{fci_error_millihartree:16.8e}  "
                f"{alpha:12.8f}  {beta_next:12.8f}  "
                f"{row['overlap_condition']:18.8e}"
            )
        condition_data.append(
            (
                epsilon,
                [row["K"] for row in result.history],
                [row["overlap_condition"] for row in result.history],
            )
        )
        energy_data.append(
            (
                epsilon,
                [row["K"] for row in result.history],
                [row["energy_hartree"] for row in result.history],
            )
        )
        fci_error_data.append(
            (
                epsilon,
                [row["K"] for row in result.history],
                [
                    abs(row["energy_hartree"] - fci_energy)
                    for row in result.history
                ],
            )
        )

    figure, axes = plt.subplots()
    for epsilon, dimensions, conditions in condition_data:
        excess = [(condition - 1.0) * 1e8 for condition in conditions]
        axes.plot(dimensions, excess, marker="o", label=f"$\\epsilon={epsilon:g}$")
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel(r"Condition-number excess, $(\kappa - 1) \times 10^8$")
    axes.set_title(r"H4 overlap-matrix condition number ($\kappa$)")
    axes.grid(True, linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(CONDITION_PLOT_FILE, dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots()
    for epsilon, dimensions, energies in energy_data:
        axes.plot(dimensions, energies, marker="o", label=f"$\\epsilon={epsilon:g}$")
    axes.axhline(
        fci_energy,
        color="black",
        linestyle="--",
        label=f"FCI: {fci_energy:.6f} Ha",
    )
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel("Ground-state energy estimate (Hartree)")
    axes.set_title("H4 energy convergence")
    axes.grid(True, linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(ENERGY_PLOT_FILE, dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots()
    for epsilon, dimensions, errors in fci_error_data:
        axes.semilogy(
            dimensions,
            [error * 1000 for error in errors],
            marker="o",
            label=f"$\\epsilon={epsilon:g}$",
        )
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel("Absolute energy error from FCI (milliHartree)")
    axes.set_title("H4 energy error relative to active-space FCI")
    axes.grid(True, which="both", linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(FCI_ERROR_PLOT_FILE, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        with redirect_stdout(output):
            main()
