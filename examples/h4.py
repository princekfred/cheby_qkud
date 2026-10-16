"""Run the OCQKUDSIN sine filter on square H4."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pennylane as qml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocqkudsin import (
    MolecularProblem,
    QuantumOCQKUDSIN,
    build_molecular_problem,
    exact_sector_energy,
)


GEOMETRY_ANGSTROM = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5],
        [0.0, 1.5, 0.0],
        [0.0, 1.5, 1.5],
    ]
)
EPSILONS = (0.5, 0.8, 1.2)
MAX_DIMENSION = 8
ARCHIVE_FILE = Path(__file__).resolve().parent / "data" / "h4_square_hamiltonian.npz"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "ocqkudsin"
OUTPUT_FILE = OUTPUT_DIR / "output.txt"
CONDITION_PLOT_FILE = OUTPUT_DIR / "condition_number.png"
ENERGY_PLOT_FILE = OUTPUT_DIR / "energy_convergence.png"
FCI_ERROR_PLOT_FILE = OUTPUT_DIR / "energy_error_fci.png"


def _pauli_operator(word: str):
    if word == "I()":
        return qml.Identity(0)
    gates = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    factors = []
    for factor in word.split(" @ "):
        match = re.fullmatch(r"([XYZ])\((\d+)\)", factor)
        if match is None:
            raise ValueError(f"unsupported archived Pauli word: {word}")
        gate, wire = match.groups()
        factors.append(gates[gate](int(wire)))
    return qml.prod(*factors) if len(factors) > 1 else factors[0]


def load_archived_problem(snapshot: Path = ARCHIVE_FILE) -> MolecularProblem:
    """Load the fixed H4 Hamiltonian used by the ocqkudsin comparison."""

    with np.load(snapshot, allow_pickle=False) as saved:
        matrix = np.asarray(saved["H"], dtype=complex)
        coefficients = np.asarray(saved["pauli_coefficients"], dtype=float)
        words = [str(word) for word in saved["pauli_words"]]
        hf_state = np.asarray(saved["hf_state"], dtype=complex)
        pauli_bound = float(saved["pauli_bound"])
        geometry = np.asarray(saved["geometry_angstrom"], dtype=float)

    if not np.array_equal(geometry, GEOMETRY_ANGSTROM):
        raise ValueError("the archived geometry is not the square H4 example")
    n_qubits = int(round(np.log2(len(hf_state))))
    if matrix.shape != (2**n_qubits, 2**n_qubits):
        raise ValueError("the archived Hamiltonian has the wrong shape")
    if not np.isclose(np.sum(np.abs(coefficients)), pauli_bound, atol=1e-9):
        raise ValueError("the archived Pauli bound disagrees with its coefficients")
    operators = tuple(_pauli_operator(word) for word in words)
    hamiltonian = qml.dot(coefficients, operators)
    reconstructed = np.asarray(
        qml.matrix(hamiltonian, wire_order=range(n_qubits)), dtype=complex
    )
    if not np.allclose(reconstructed, matrix, rtol=0, atol=1e-10):
        raise ValueError("the archived Pauli terms disagree with its matrix")
    hf_index = int(np.argmax(np.abs(hf_state)))
    if not np.isclose(abs(hf_state[hf_index]), 1.0) or np.count_nonzero(hf_state) != 1:
        raise ValueError("the archived Hartree-Fock state is not a basis state")
    hf_bits = np.fromiter(
        (int(bit) for bit in f"{hf_index:0{n_qubits}b}"), dtype=int
    )
    return MolecularProblem(
        hamiltonian=hamiltonian,
        coefficients=coefficients,
        operators=operators,
        n_qubits=n_qubits,
        hf_bits=hf_bits,
        hf_state=hf_state,
        pauli_bound=pauli_bound,
    )


def _load_problem(rebuild: bool) -> tuple[MolecularProblem, str]:
    if rebuild:
        problem = build_molecular_problem(
            symbols=["H", "H", "H", "H"],
            geometry_angstrom=GEOMETRY_ANGSTROM,
            active_electrons=4,
            active_orbitals=4,
            basis="sto-3g",
        )
        source = f"fresh PennyLane {qml.__version__} molecular Hamiltonian"
    else:
        problem = load_archived_problem()
        source = "bundled H4 Hamiltonian from the ocqkudsin comparison"
    return problem, source


def main(
    *,
    rebuild: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem, source = _load_problem(rebuild)

    fci_energy = exact_sector_energy(problem, n_alpha=2, n_beta=2)
    print("Square H4, side length 1.5 Angstrom, STO-3G")
    print("Solver: OCQKUDSIN")
    print(f"Hamiltonian source: {source}")
    print(f"Pauli coefficient bound: {problem.pauli_bound:.12f} Hartree")
    print(f"FCI energy (active space, 2 alpha + 2 beta): {fci_energy:.12f} Hartree")
    condition_data = []
    energy_data = []
    fci_error_data = []
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
    figure.savefig(output_dir / FCI_ERROR_PLOT_FILE.name, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="generate a new Hamiltonian with the installed PennyLane version",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for OCQKUDSIN results (default: output/ocqkudsin/)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / OUTPUT_FILE.name
    with output_file.open("w", encoding="utf-8") as output:
        with redirect_stdout(output):
            main(rebuild=args.rebuild, output_dir=output_dir)
    print(f"OCQKUDSIN results: {output_file}")
