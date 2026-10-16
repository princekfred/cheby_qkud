"""Build linear H4 and run the OCQKUDSIN sine filter."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocqkudsin import (
    QuantumOCQKUDSIN,
    build_molecular_problem,
    exact_sector_energy,
)


GEOMETRY_ANGSTROM = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 6.0],
        [0.0, 0.0, 9.0],
    ]
)
EPSILONS = (0.5, 0.8, 1.2)
MAX_DIMENSION = 8
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "h4_lin"
OUTPUT_FILE = OUTPUT_DIR / "output.txt"
CONDITION_PLOT_FILE = OUTPUT_DIR / "condition_number.png"
ENERGY_PLOT_FILE = OUTPUT_DIR / "energy_convergence.png"
FCI_ERROR_PLOT_FILE = OUTPUT_DIR / "energy_error_fci.png"
LANCZOS_PLOT_FILE = OUTPUT_DIR / "lanczos_coefficients.png"
CHEBYSHEV_GROWTH_PLOT_FILE = OUTPUT_DIR / "chebyshev_coefficient_growth.png"
CHEBYSHEV_HEATMAP_FILE = OUTPUT_DIR / "chebyshev_coefficient_heatmap.png"
CHEBYSHEV_DATA_FILE = OUTPUT_DIR / "chebyshev_coefficients.csv"


def _save_chebyshev_coefficients(
    output_dir: Path, coefficient_data: list[tuple[float, list[np.ndarray]]]
) -> None:
    """Save every c[n,m] and plot its magnitude across Lanczos iterations."""

    with (output_dir / CHEBYSHEV_DATA_FILE.name).open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(("epsilon", "n", "m", "real", "imaginary", "magnitude"))
        for epsilon, rows in coefficient_data:
            for n, coefficients in enumerate(rows):
                for m, value in enumerate(coefficients):
                    writer.writerow(
                        (
                            f"{epsilon:g}",
                            n,
                            m,
                            f"{value.real:.17g}",
                            f"{value.imag:.17g}",
                            f"{abs(value):.17g}",
                        )
                    )

    max_iterations = max(len(rows) for _, rows in coefficient_data)
    max_order = max(len(row) - 1 for _, rows in coefficient_data for row in rows)
    logged_magnitudes = [
        np.log10(abs(value))
        for _, rows in coefficient_data
        for row in rows
        for value in row
        if abs(value) > 0
    ]
    color_min = min(0.0, float(np.floor(min(logged_magnitudes))))
    color_max = max(color_min + 1.0, float(np.ceil(max(logged_magnitudes))))

    figure, axes = plt.subplots(
        1, len(coefficient_data), figsize=(15, 5.5), sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("white")
    for axes_one, (epsilon, rows) in zip(axes, coefficient_data):
        values = np.full((max_iterations, max_order + 1), np.nan)
        for n, coefficients in enumerate(rows):
            for m, value in enumerate(coefficients):
                if abs(value) > 0:
                    values[n, m] = np.log10(abs(value))
        image = axes_one.imshow(
            np.ma.masked_invalid(values),
            origin="lower", aspect="equal", cmap=colormap,
            vmin=color_min, vmax=color_max,
        )
        for n, coefficients in enumerate(rows):
            for m, _value in enumerate(coefficients):
                log_value = values[n, m]
                if np.isfinite(log_value):
                    fraction = (log_value - color_min) / (color_max - color_min)
                    axes_one.text(
                        m, n, f"{log_value:.1f}", ha="center", va="center",
                        color="white" if fraction < 0.55 else "black", fontsize=8,
                    )
        axes_one.set_title(rf"$\epsilon={epsilon:g}$")
        axes_one.set_xticks(range(max_order + 1))
        axes_one.set_yticks(range(max_iterations))
        axes_one.set_xlabel(r"Chebyshev order $m$")
    axes[0].set_ylabel(r"Lanczos state $q_n$ (iteration $n$)")
    figure.suptitle(r"Linear H$_4$: $\log_{10}|c_{n,m}|$ at each iteration")
    figure.colorbar(
        image, ax=axes.tolist(), shrink=0.83,
        label=r"$\log_{10}|c_{n,m}|$",
    )
    figure.savefig(output_dir / CHEBYSHEV_HEATMAP_FILE.name, dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(coefficient_data), figsize=(15, 5.5), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for axes_one, (epsilon, rows) in zip(axes, coefficient_data):
        for m in range(max_order + 1):
            iterations = [n for n, row in enumerate(rows) if m < len(row)]
            if not iterations:
                continue
            log_values = [
                np.log10(abs(rows[n][m])) if abs(rows[n][m]) > 0 else np.nan
                for n in iterations
            ]
            axes_one.plot(
                iterations, log_values, marker="o", markersize=4,
                linewidth=1.5, color=f"C{m}", label=rf"$m={m}$",
            )
        axes_one.set_title(rf"$\epsilon={epsilon:g}$")
        axes_one.set_xticks(range(max_iterations))
        axes_one.set_xlabel(r"Lanczos iteration $n$")
        axes_one.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel(r"$\log_{10}|c_{n,m}|$")
    figure.suptitle(r"Linear H$_4$: growth of each Chebyshev coefficient")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=max_order + 1, loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.94))
    figure.savefig(output_dir / CHEBYSHEV_GROWTH_PLOT_FILE.name, dpi=200)
    plt.close(figure)


def main(
    *,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = build_molecular_problem(
        symbols=["H", "H", "H", "H"],
        geometry_angstrom=GEOMETRY_ANGSTROM,
        active_electrons=4,
        active_orbitals=4,
        basis="sto-3g",
    )

    fci_energy = exact_sector_energy(problem, n_alpha=2, n_beta=2)
    print("Linear H4, nearest-neighbor spacing 3.0 Angstrom, STO-3G")
    print("Solver: OCQKUDSIN")
    print("Hamiltonian source: built from the linear H4 geometry with PennyLane")
    print(f"Pauli coefficient bound: {problem.pauli_bound:.12f} Hartree")
    print(f"FCI energy (active space, 2 alpha + 2 beta): {fci_energy:.12f} Hartree")
    condition_data = []
    energy_data = []
    fci_error_data = []
    coefficient_data = []
    chebyshev_data = []
    for epsilon in EPSILONS:
        result = QuantumOCQKUDSIN(
            problem,
            epsilon=epsilon,
            max_dimension=MAX_DIMENSION,
            # At K=8 the Lanczos moment difference can cancel to about 1e-10.
            moment_tolerance=1e-9,
        ).run()

        print(f"\nepsilon = {epsilon:g}")
        print(
            " K   Energy (Hartree)       FCI error (Ha)       alpha"
            "          beta      overlap condition"
        )
        for row in result.history:
            k = row["K"]
            alpha = result.alpha[k - 1]
            beta_next = row["beta_next"]
            fci_error_hartree = abs(row["energy_hartree"] - fci_energy)
            print(
                f"{k:2d}  {row['energy_hartree']:16.12f}  "
                f"{fci_error_hartree:16.8e}  "
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
        coefficient_data.append((epsilon, result.alpha.copy(), result.beta.copy()))
        chebyshev_data.append(
            (
                epsilon,
                [
                    coefficients.copy()
                    for coefficients in result.chebyshev_coefficients
                ],
            )
        )

    figure, axes = plt.subplots()
    for epsilon, dimensions, conditions in condition_data:
        axes.plot(dimensions, conditions, marker="o", label=f"$\\epsilon={epsilon:g}$")
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel(r"Overlap-matrix condition number, $\kappa(S)$")
    axes.set_title(r"H4 overlap-matrix condition number ($\kappa$)")
    axes.yaxis.set_major_formatter(FormatStrFormatter("%.8f"))
    axes.grid(True, linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_dir / CONDITION_PLOT_FILE.name, dpi=200)
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
    figure.savefig(output_dir / ENERGY_PLOT_FILE.name, dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots()
    for epsilon, dimensions, errors in fci_error_data:
        axes.semilogy(
            dimensions,
            errors,
            marker="o",
            label=f"$\\epsilon={epsilon:g}$",
        )
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel("Absolute energy error from FCI (Ha)")
    axes.set_title("H4 energy error relative to active-space FCI")
    axes.grid(True, which="both", linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_dir / FCI_ERROR_PLOT_FILE.name, dpi=200)
    plt.close(figure)

    figure, (alpha_axes, beta_axes) = plt.subplots(
        2, 1, figsize=(8.5, 6.5), sharex=True
    )
    for index, (epsilon, alpha, beta) in enumerate(coefficient_data):
        color = f"C{index}"
        alpha_axes.plot(
            np.arange(1, len(alpha) + 1),
            alpha,
            color=color,
            marker="o",
            label=rf"$\epsilon={epsilon:g}$",
        )
        beta_axes.plot(
            np.arange(1, len(beta)),
            beta[1:],
            color=color,
            marker="o",
        )
    alpha_axes.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    alpha_axes.set_ylabel(r"Diagonal coefficient $\alpha_{K-1}$")
    beta_axes.set_ylabel(r"Off-diagonal coefficient $\beta_K$")
    beta_axes.set_xlabel(r"Lanczos iteration $K$")
    max_iterations = max(len(alpha) for _, alpha, _ in coefficient_data)
    beta_axes.set_xticks(range(1, max_iterations + 1))
    for axes in (alpha_axes, beta_axes):
        axes.grid(True, linestyle="--", alpha=0.4)
    alpha_axes.legend(ncol=len(coefficient_data))
    figure.suptitle(r"Linear H$_4$ OCQKUDSIN Lanczos coefficients")
    figure.text(
        0.5,
        0.025,
        r"Only $\beta_K$ values used to form the next state are shown.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output_dir / LANCZOS_PLOT_FILE.name, dpi=200)
    plt.close(figure)

    _save_chebyshev_coefficients(output_dir, chebyshev_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for OCQKUDSIN results (default: output/h4_lin/)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / OUTPUT_FILE.name
    with output_file.open("w", encoding="utf-8") as output:
        with redirect_stdout(output):
            main(output_dir=output_dir)
    print(f"OCQKUDSIN results: {output_file}")
