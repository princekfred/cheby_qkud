"""Run OCQKUDSIN from a spin-adapted quintet state of neutral Fe2S2.

The planar rhombus is derived from approximate bond lengths reported for an
isolated neutral Fe2S2 singlet (J. Phys. Chem. A 2019, DOI 10.1021/acs.jpca.9b08201).
It is not a published Cartesian geometry or an optimized quintet geometry.
Restricted Hartree-Fock supplies spatial orbitals for the CAS(4e,5o)
Hamiltonian. The initial quantum state is an equal superposition of five
spin-adapted S=2, Ms=0 configuration state functions, not a Hartree-Fock
determinant.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pennylane as qml
from pyscf import ao2mo, fci, gto, mcscf, scf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocqkudsin import MolecularProblem, QuantumOCQKUDSIN


SYMBOLS = ("Fe", "Fe", "S", "S")
# Illustrative planar rhombus: Fe-Fe = 2.432 A, Fe-S = 2.131 A.
# Approximate lengths derive from DOI 10.1021/acs.jpca.9b08201; they are not
# optimized coordinates for the bare 6-31G/RHF model used here.
GEOMETRY_ANGSTROM = np.array(
    [
        [-1.216, 0.0, 0.0],
        [1.216, 0.0, 0.0],
        [0.0, -1.750, 0.0],
        [0.0, 1.750, 0.0],
    ],
    dtype=float,
)
BASIS = "6-31g"
CHARGE = 0
ORBITAL_REFERENCE_SPIN = 0
ACTIVE_ELECTRONS = 4
ACTIVE_ORBITALS = 5
EPSILONS = (0.5, 0.8, 1.2)
MAX_DIMENSION = 5
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "fe2s2_quintet"


def _quintet_csf_basis() -> np.ndarray:
    """Return the five S=2, Ms=0 CSFs for four electrons in five orbitals."""

    n_qubits = 2 * ACTIVE_ORBITALS
    basis = np.zeros((2**n_qubits, ACTIVE_ORBITALS), dtype=complex)
    for empty_orbital in range(ACTIVE_ORBITALS):
        occupied = tuple(
            orbital for orbital in range(ACTIVE_ORBITALS)
            if orbital != empty_orbital
        )
        for alpha_orbitals in combinations(occupied, 2):
            bits = np.zeros(n_qubits, dtype=int)
            for orbital in occupied:
                bits[2 * orbital + (0 if orbital in alpha_orbitals else 1)] = 1
            basis[int("".join(map(str, bits)), 2), empty_orbital] = 1 / np.sqrt(6)
    return basis


def _rhf_orbitals(geometry_angstrom: np.ndarray):
    """Converge restricted orbitals used to build the active Hamiltonian."""

    molecule = gto.M(
        atom=list(zip(SYMBOLS, geometry_angstrom)),
        unit="Angstrom",
        basis=BASIS,
        charge=CHARGE,
        spin=ORBITAL_REFERENCE_SPIN,
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.max_cycle = 150
    mean_field.conv_tol = 1e-10
    mean_field.conv_tol_grad = 1e-7
    energy = float(mean_field.kernel())
    if not mean_field.converged or not np.isfinite(energy):
        raise RuntimeError(
            "Fe2S2 RHF failed to converge; no Hamiltonian was constructed "
            "from unconverged orbitals."
        )
    return mean_field


def _build_shifted_problem(
    geometry_angstrom: np.ndarray,
) -> tuple[MolecularProblem, float, float, float]:
    """Remove the scalar identity energy to resolve the sine-filter moments.

    The scalar shift leaves the molecular eigenvectors and energy differences
    unchanged. It is added back to every reported physical energy.
    """

    mean_field = _rhf_orbitals(geometry_angstrom)
    rhf_energy = float(mean_field.e_tot)
    casci = mcscf.CASCI(mean_field, ACTIVE_ORBITALS, ACTIVE_ELECTRONS)
    one_mo, core_energy = casci.get_h1eff()
    two_chemist = ao2mo.restore(
        1, casci.get_h2eff(), ACTIVE_ORBITALS
    )
    quintet_fci_energy, _ = fci.direct_spin1.kernel(
        one_mo, two_chemist, ACTIVE_ORBITALS,
        (ACTIVE_ELECTRONS, 0),
        ecore=core_energy,
    )
    two_physicist = np.swapaxes(two_chemist, 1, 3)
    fermionic = qml.qchem.fermionic_observable(
        np.array([core_energy]), one_mo, two_physicist
    )
    hamiltonian = qml.simplify(
        qml.jordan_wigner(fermionic, ps=True, tol=1e-10).hamiltonian()
    )
    n_qubits = 2 * ACTIVE_ORBITALS
    coefficients, operators = hamiltonian.terms()
    coefficients = np.asarray(coefficients, dtype=complex)
    if np.max(np.abs(coefficients.imag)) > 1e-10:
        raise ValueError("the Fe2S2 Pauli Hamiltonian has nonreal coefficients")
    coefficients = coefficients.real

    identity_indices = [
        index for index, operator in enumerate(operators)
        if operator.name == "Identity"
    ]
    if not identity_indices:
        raise ValueError("the Fe2S2 Hamiltonian has no identity energy to shift")
    identity_shift = float(np.sum(coefficients[identity_indices]))
    nonidentity_indices = [
        index for index in range(len(operators)) if index not in identity_indices
    ]
    shifted_coefficients = coefficients[nonidentity_indices]
    shifted_operators = tuple(operators[index] for index in nonidentity_indices)
    shifted_hamiltonian = qml.Hamiltonian(
        shifted_coefficients, list(shifted_operators)
    )
    pauli_bound = float(np.sum(np.abs(shifted_coefficients)))
    if not np.isfinite(pauli_bound) or pauli_bound <= 0:
        raise ValueError("the shifted Pauli coefficient bound must be positive")

    # The solver's legacy hf_state field holds its arbitrary input vector.
    # This equal combination spans all five quintet configurations; hf_bits
    # is unused because no single determinant describes the initial state.
    quintet_state = np.sum(_quintet_csf_basis(), axis=1) / np.sqrt(ACTIVE_ORBITALS)
    problem = MolecularProblem(
        hamiltonian=shifted_hamiltonian,
        coefficients=shifted_coefficients,
        operators=shifted_operators,
        n_qubits=n_qubits,
        hf_bits=np.array([], dtype=int),
        hf_state=quintet_state,
        pauli_bound=pauli_bound,
    )
    return problem, identity_shift, rhf_energy, float(quintet_fci_energy)


def _quintet_connected_exact_energy(
    problem: MolecularProblem, identity_shift: float, fci_energy: float
) -> tuple[float, float]:
    """Find the lowest quintet eigenstate reached by the CSF reference."""

    basis = _quintet_csf_basis()
    matrix = np.asarray(
        qml.matrix(problem.hamiltonian, wire_order=range(problem.n_qubits)),
        dtype=complex,
    )
    projected = basis.conj().T @ matrix @ basis
    if np.linalg.norm(matrix @ basis - basis @ projected) > 1e-7:
        raise ValueError("the spin-adapted quintet subspace is not invariant")
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    if abs(eigenvalues[0] + identity_shift - fci_energy) > 1e-7:
        raise ValueError("the Pauli Hamiltonian disagrees with quintet CASCI FCI")
    weights = np.abs(eigenvectors.conj().T @ basis.conj().T @ problem.hf_state) ** 2
    connected = np.flatnonzero(weights > 1e-10)
    if not len(connected):
        raise ValueError("no exact quintet state overlaps the initial state")
    return float(eigenvalues[connected[0]] + identity_shift), float(weights[0])


def _save_chebyshev_coefficients(
    output_dir: Path, data: list[tuple[float, list[np.ndarray]]]
) -> None:
    """Save all c[n,m] values and their iteration-by-iteration magnitude plots."""

    with (output_dir / "chebyshev_coefficients.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(("epsilon", "n", "m", "real", "imaginary", "magnitude"))
        for epsilon, rows in data:
            for n, row in enumerate(rows):
                for m, value in enumerate(row):
                    writer.writerow(
                        (epsilon, n, m, value.real, value.imag, abs(value))
                    )

    max_n = max(len(rows) for _, rows in data)
    max_m = max(len(row) for _, rows in data for row in rows)
    figure, axes = plt.subplots(
        1, len(data), figsize=(15, 5.5), sharex=True, sharey=True
    )
    for axes_one, (epsilon, rows) in zip(np.atleast_1d(axes), data):
        for m in range(max_m):
            n_values = [n for n, row in enumerate(rows) if m < len(row)]
            if n_values:
                log_values = [
                    np.log10(abs(rows[n][m])) if abs(rows[n][m]) else np.nan
                    for n in n_values
                ]
                axes_one.plot(
                    n_values, log_values, marker="o", markersize=4,
                    label=rf"$m={m}$",
                )
        axes_one.set_title(rf"$\epsilon={epsilon:g}$")
        axes_one.set_xticks(range(max_n))
        axes_one.set_xlabel(r"Lanczos iteration $n$")
        axes_one.grid(True, linestyle="--", alpha=0.35)
    np.atleast_1d(axes)[0].set_ylabel(r"$\log_{10}|c_{n,m}|$")
    figure.suptitle(r"Fe$_2$S$_2$: growth of each Chebyshev coefficient")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=max_m, loc="lower center")
    figure.tight_layout(rect=(0, 0.1, 1, 0.94))
    figure.savefig(output_dir / "chebyshev_coefficient_growth.png", dpi=200)
    plt.close(figure)

    arrays = []
    for _, rows in data:
        values = np.full((max_n, max_m), np.nan)
        for n, row in enumerate(rows):
            for m, value in enumerate(row):
                if abs(value):
                    values[n, m] = np.log10(abs(value))
        arrays.append(values)
    finite = np.concatenate([values[np.isfinite(values)] for values in arrays])
    color_min = min(0.0, float(np.floor(np.min(finite))))
    color_max = max(color_min + 1.0, float(np.ceil(np.max(finite))))
    figure, axes = plt.subplots(
        1, len(data), figsize=(15, 5.5), sharey=True, constrained_layout=True
    )
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("white")
    for axes_one, (epsilon, _), values in zip(
        np.atleast_1d(axes), data, arrays
    ):
        image = axes_one.imshow(
            np.ma.masked_invalid(values), origin="lower", aspect="equal",
            cmap=colormap, vmin=color_min, vmax=color_max,
        )
        axes_one.set_title(rf"$\epsilon={epsilon:g}$")
        axes_one.set_xticks(range(max_m))
        axes_one.set_yticks(range(max_n))
        axes_one.set_xlabel(r"Chebyshev order $m$")
    np.atleast_1d(axes)[0].set_ylabel(r"Lanczos iteration $n$")
    figure.suptitle(r"Fe$_2$S$_2$: $\log_{10}|c_{n,m}|$ at each iteration")
    figure.colorbar(
        image, ax=np.atleast_1d(axes).tolist(), shrink=0.83,
        label=r"$\log_{10}|c_{n,m}|$",
    )
    figure.savefig(output_dir / "chebyshev_coefficient_heatmap.png", dpi=200)
    plt.close(figure)


def main(*, output_dir: Path = OUTPUT_DIR) -> None:
    """Build the Fe2S2 Hamiltonian directly and save OCQKUDSIN results."""

    output_dir = Path(output_dir)
    geometry = np.asarray(GEOMETRY_ANGSTROM, dtype=float)
    if geometry.shape != (len(SYMBOLS), 3) or not np.all(np.isfinite(geometry)):
        raise ValueError("GEOMETRY_ANGSTROM must contain four finite XYZ rows")
    problem, identity_shift, rhf_energy, fci_energy = _build_shifted_problem(geometry)
    connected_energy, fci_weight = _quintet_connected_exact_energy(
        problem, identity_shift, fci_energy
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fe2S2 (2Fe-2S), neutral, 6-31G, 4 electrons in 5 active orbitals")
    print("Geometry: illustrative planar rhombus; not optimized for the quintet")
    print("Initial state: spin-adapted quintet (S=2, Ms=0)")
    print("Equal superposition of five four-singly-occupied CSFs (30 determinants).")
    for symbol, coordinates in zip(SYMBOLS, geometry):
        print(f"  {symbol:2s}  {coordinates[0]: .6f}  {coordinates[1]: .6f}  {coordinates[2]: .6f} Angstrom")
    print("Hamiltonian source: PySCF RHF orbital basis and 4e/5o CASCI integrals")
    print(f"RHF orbital calculation converged; energy = {rhf_energy:.12f} Ha")
    print(f"Identity energy offset: {identity_shift:.12f} Ha")
    print(f"Shifted Pauli coefficient bound: {problem.pauli_bound:.12f} Ha")
    print("Sine filter acts on the identity-shifted Hamiltonian; the offset is restored below.")
    print(f"Quintet active-space FCI (4 alpha + 0 beta): {fci_energy:.12f} Ha")
    print("The Ms=2 FCI sector has the same energy spectrum as the Ms=0 quintet sector.")
    print(f"Lowest initial-state-connected quintet energy: {connected_energy:.12f} Ha")
    print(f"Initial-state weight in the quintet FCI ground state: {fci_weight:.3e}")
    if fci_weight <= 1e-10:
        print("The quintet FCI ground state is inaccessible from this symmetry-preserving initial state.")

    condition_data = []
    energy_data = []
    error_data = []
    lanczos_data = []
    chebyshev_data = []
    for epsilon in EPSILONS:
        result = QuantumOCQKUDSIN(
            problem,
            epsilon=epsilon,
            max_dimension=MAX_DIMENSION,
            moment_tolerance=1e-9,
        ).run()
        print(f"\nepsilon = {epsilon:g}")
        print(
            " K   Energy (Ha)      Quintet FCI error (Ha)  alpha"
            "          beta      overlap condition"
        )
        dimensions = []
        conditions = []
        energies = []
        errors = []
        for row in result.history:
            k = row["K"]
            energy = row["energy_hartree"] + identity_shift
            error = abs(energy - fci_energy)
            print(
                f"{k:2d}  {energy:18.12f}  {error:16.8e}  "
                f"{result.alpha[k - 1]:12.8f}  {row['beta_next']:12.8f}  "
                f"{row['overlap_condition']:18.8e}"
            )
            dimensions.append(k)
            conditions.append(row["overlap_condition"])
            energies.append(energy)
            errors.append(error)
        condition_data.append((epsilon, dimensions, conditions))
        energy_data.append((epsilon, dimensions, energies))
        error_data.append((epsilon, dimensions, errors))
        lanczos_data.append((epsilon, result.alpha.copy(), result.beta.copy()))
        chebyshev_data.append(
            (epsilon, [row.copy() for row in result.chebyshev_coefficients])
        )

    figure, axes = plt.subplots()
    for epsilon, dimensions, conditions in condition_data:
        axes.plot(dimensions, conditions, marker="o", label=rf"$\epsilon={epsilon:g}$")
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel(r"Overlap-matrix condition number, $\kappa(S)$")
    axes.set_title(r"Fe$_2$S$_2$ overlap-matrix condition number")
    axes.yaxis.set_major_formatter(FormatStrFormatter("%.8f"))
    axes.grid(True, linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "condition_number.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots()
    for epsilon, dimensions, energies in energy_data:
        axes.plot(dimensions, energies, marker="o", label=rf"$\epsilon={epsilon:g}$")
    axes.axhline(
        fci_energy, color="black", linestyle="--",
        label=f"Quintet FCI: {fci_energy:.6f} Ha",
    )
    if connected_energy - fci_energy > 1e-8:
        axes.axhline(
            connected_energy, color="gray", linestyle=":",
            label=f"Initial-state-connected: {connected_energy:.6f} Ha",
        )
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel("Quintet energy estimate (Ha)")
    axes.set_title(r"Fe$_2$S$_2$ quintet energy convergence")
    axes.grid(True, linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "energy_convergence.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots()
    for epsilon, dimensions, errors in error_data:
        axes.semilogy(
            dimensions, np.maximum(errors, np.finfo(float).tiny),
            marker="o", label=rf"$\epsilon={epsilon:g}$",
        )
    if connected_energy - fci_energy > 1e-8:
        axes.axhline(
            connected_energy - fci_energy, color="gray", linestyle=":",
            label="Initial-state-connected error floor",
        )
    axes.set_xlabel("Krylov dimension (K)")
    axes.set_ylabel("Absolute error from quintet FCI (Ha)")
    axes.set_title(r"Fe$_2$S$_2$ energy error relative to quintet FCI")
    axes.grid(True, which="both", linestyle="--", alpha=0.5)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "energy_error_fci.png", dpi=200)
    plt.close(figure)

    figure, (alpha_axes, beta_axes) = plt.subplots(
        2, 1, figsize=(8.5, 6.5), sharex=True
    )
    for index, (epsilon, alpha, beta) in enumerate(lanczos_data):
        color = f"C{index}"
        alpha_axes.plot(
            np.arange(1, len(alpha) + 1), alpha, color=color,
            marker="o", label=rf"$\epsilon={epsilon:g}$",
        )
        beta_axes.plot(
            np.arange(1, len(beta)), beta[1:], color=color, marker="o"
        )
    alpha_axes.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    alpha_axes.set_ylabel(r"Diagonal coefficient $\alpha_{K-1}$")
    beta_axes.set_ylabel(r"Off-diagonal coefficient $\beta_K$")
    beta_axes.set_xlabel(r"Lanczos iteration $K$")
    beta_axes.set_xticks(range(1, max(len(a) for _, a, _ in lanczos_data) + 1))
    for axes in (alpha_axes, beta_axes):
        axes.grid(True, linestyle="--", alpha=0.4)
    alpha_axes.legend(ncol=len(lanczos_data))
    figure.suptitle(r"Fe$_2$S$_2$ OCQKUDSIN Lanczos coefficients")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_dir / "lanczos_coefficients.png", dpi=200)
    plt.close(figure)

    _save_chebyshev_coefficients(output_dir, chebyshev_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="directory for OCQKUDSIN results (default: output/fe2s2_quintet/)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.txt"
    with output_file.open("w", encoding="utf-8") as output:
        with redirect_stdout(output):
            main(output_dir=output_dir)
    print(f"OCQKUDSIN results: {output_file}")
