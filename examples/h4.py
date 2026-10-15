"""Run square H4 with OCQKUD at epsilon=0.8 and 1.2."""

from __future__ import annotations

import numpy as np

from ocqkud import QuantumOCQKUD, build_molecular_problem


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


def main() -> None:
    problem = build_molecular_problem(
        symbols=["H", "H", "H", "H"],
        geometry_angstrom=GEOMETRY_ANGSTROM,
        active_electrons=4,
        active_orbitals=4,
        basis="sto-3g",
    )

    print("Square H4, side length 1.5 Angstrom, STO-3G")
    print(f"Pauli coefficient bound: {problem.pauli_bound:.12f} Hartree")
    for epsilon in EPSILONS:
        result = QuantumOCQKUD(
            problem,
            epsilon=epsilon,
            max_dimension=MAX_DIMENSION,
        ).run()

        print(f"\nepsilon = {epsilon:g}")
        print(" K   Energy (Hartree)      alpha          beta")
        for row in result.history:
            k = row["K"]
            alpha = result.alpha[k - 1]
            beta_next = row["beta_next"]
            print(
                f"{k:2d}  {row['energy_hartree']:16.12f}  "
                f"{alpha:12.8f}  {beta_next:12.8f}"
            )


if __name__ == "__main__":
    main()
