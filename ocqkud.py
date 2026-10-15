"""Small-system circuit implementation of OCQKUD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pennylane as qml


Array = np.ndarray


@dataclass(frozen=True)
class MolecularProblem:
    """Qubit Hamiltonian and Hartree-Fock reference state."""

    hamiltonian: object
    coefficients: Array
    operators: tuple
    n_qubits: int
    hf_bits: Array
    hf_state: Array
    pauli_bound: float


@dataclass
class OCQKUDResult:
    """Solver outputs for one epsilon value."""

    epsilon: float
    scale: float
    alpha: Array
    second_moments: Array
    beta: Array
    chebyshev_coefficients: list[Array]
    state_vectors: list[Array]
    overlap: Array
    projected_hamiltonian: Array
    physical_energies: Array
    ritz_coefficients: Array
    history: list[dict]


def build_molecular_problem(
    symbols: Sequence[str],
    geometry_angstrom: Array,
    *,
    active_electrons: int,
    active_orbitals: int,
    basis: str = "sto-3g",
    charge: int = 0,
) -> MolecularProblem:
    """Construct a Jordan-Wigner Hamiltonian and Hartree-Fock state."""

    geometry = np.asarray(geometry_angstrom, dtype=float)
    if geometry.shape != (len(symbols), 3):
        raise ValueError("geometry must have shape (number of atoms, 3)")
    if active_electrons <= 0 or active_electrons % 2:
        raise ValueError("active_electrons must be a positive even integer")
    if active_orbitals < active_electrons // 2:
        raise ValueError("active_orbitals is too small for active_electrons")

    # PennyLane's built-in basis list omits STO-6G; its basis-set-exchange
    # integration supplies the same basis data when load_data is enabled.
    molecule = qml.qchem.Molecule(
        list(symbols),
        geometry,
        charge=charge,
        basis_name=basis,
        load_data=basis.lower() == "sto-6g",
        unit="angstrom",
    )
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule,
        method="dhf",
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping="jordan_wigner",
    )
    coefficients, operators = hamiltonian.terms()
    coefficients = np.real_if_close(
        np.asarray(coefficients, dtype=complex)
    ).astype(float)
    scale = float(np.sum(np.abs(coefficients)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("the Pauli coefficient bound must be positive")

    hf_bits = np.asarray(qml.qchem.hf_state(active_electrons, n_qubits), dtype=int)
    hf_index = int("".join(str(bit) for bit in hf_bits), 2)
    hf_state = np.zeros(2**n_qubits, dtype=complex)
    hf_state[hf_index] = 1.0
    return MolecularProblem(
        hamiltonian=hamiltonian,
        coefficients=coefficients,
        operators=tuple(operators),
        n_qubits=n_qubits,
        hf_bits=hf_bits,
        hf_state=hf_state,
        pauli_bound=scale,
    )


def _state_preparation_unitary(state: Array) -> Array:
    """Return a Householder unitary that prepares ``state`` up to global phase."""

    target = np.asarray(state, dtype=complex)
    target = target / np.linalg.norm(target)
    if abs(target[0]) > 1e-15:
        target = np.exp(-1j * np.angle(target[0])) * target
    origin = np.zeros_like(target)
    origin[0] = 1.0
    difference = origin - target
    norm = np.linalg.norm(difference)
    if norm < 1e-14:
        return np.eye(len(target), dtype=complex)
    direction = difference / norm
    unitary = np.eye(len(target), dtype=complex) - 2.0 * np.outer(
        direction, direction.conj()
    )
    if not np.allclose(unitary.conj().T @ unitary, np.eye(len(target)), atol=1e-10):
        raise FloatingPointError("state-preparation matrix is not unitary")
    return unitary


def _next_chebyshev_coefficients(
    previous: Optional[Array],
    current: Array,
    alpha: float,
    beta: float,
    beta_next: float,
) -> Array:
    """Apply the Lanczos recurrence in the Chebyshev basis."""

    degree = len(current) - 1
    result = np.zeros(degree + 2, dtype=complex)
    for m in range(degree + 2):
        if m == 1:
            result[m] += current[0]
        if m - 1 >= 1:
            result[m] += 0.5 * current[m - 1]
        if m + 1 <= degree:
            result[m] += 0.5 * current[m + 1]
        if m <= degree:
            result[m] -= alpha * current[m]
        if previous is not None and m < len(previous):
            result[m] -= beta * previous[m]
    return result / beta_next


def _clenshaw_state(
    action_matrix: Array, reference_state: Array, coefficients: Array
) -> Tuple[Array, float]:
    """Evaluate and normalize a Chebyshev expansion with Clenshaw's method."""

    coefficients = np.asarray(coefficients, dtype=complex)
    zero = np.zeros_like(reference_state, dtype=complex)
    b_next = zero.copy()
    b_next2 = zero.copy()
    for degree in range(len(coefficients) - 1, 0, -1):
        current = (
            2.0 * action_matrix @ b_next
            - b_next2
            + coefficients[degree] * reference_state
        )
        b_next2, b_next = b_next, current
    if len(coefficients) == 1:
        state = coefficients[0] * reference_state
    else:
        state = (
            action_matrix @ b_next
            - b_next2
            + coefficients[0] * reference_state
        )
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm == 0:
        raise FloatingPointError("Clenshaw recurrence produced an invalid state")
    return state / norm, norm


class QuantumOCQKUD:
    """Exact small-system circuit implementation of ``ocqku.pdf``."""

    def __init__(
        self,
        problem: MolecularProblem,
        *,
        epsilon: float,
        max_dimension: int,
        shots: Optional[int] = None,
        seed: int = 12345,
        device_name: str = "default.qubit",
        breakdown_tolerance: float = 1e-12,
        moment_tolerance: float = 1e-10,
        state_norm_tolerance: float = 1e-6,
        overlap_tolerance: float = 1e-9,
    ):
        epsilon = float(epsilon)
        if not np.isfinite(epsilon):
            raise ValueError("epsilon must be finite")
        if max_dimension < 1:
            raise ValueError("max_dimension must be positive")
        if shots is not None and shots < 1:
            raise ValueError("shots must be positive or None")

        self.problem = problem
        self.epsilon = epsilon
        self.max_dimension = int(max_dimension)
        self.shots = shots
        self.seed = int(seed)
        self.device_name = device_name
        self.breakdown_tolerance = float(breakdown_tolerance)
        self.moment_tolerance = float(moment_tolerance)
        self.state_norm_tolerance = float(state_norm_tolerance)
        self.overlap_tolerance = float(overlap_tolerance)
        self._device_counter = 0

        physical_matrix = np.asarray(
            qml.matrix(problem.hamiltonian, wire_order=range(problem.n_qubits)),
            dtype=complex,
        )
        self.hbar_matrix = (physical_matrix + physical_matrix.conj().T) / (
            2.0 * problem.pauli_bound
        )
        eigenvalues, eigenvectors = np.linalg.eigh(self.hbar_matrix)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        if spectral_radius > 1.0 + 1e-10:
            raise ValueError("the Pauli scale does not bound Hbar in [-1, 1]")

        if epsilon == 0:
            self.evolution_matrix = None
            self.double_evolution_matrix = None
            self.action_matrix = self.hbar_matrix.copy()
        else:
            phases = np.exp(-1j * epsilon * eigenvalues)
            self.evolution_matrix = (eigenvectors * phases) @ eigenvectors.conj().T
            self.double_evolution_matrix = (
                eigenvectors * np.exp(-2j * epsilon * eigenvalues)
            ) @ eigenvectors.conj().T
            filtered = np.sin(epsilon * eigenvalues) / epsilon
            self.action_matrix = (eigenvectors * filtered) @ eigenvectors.conj().T
        self.action_matrix = (
            self.action_matrix + self.action_matrix.conj().T
        ) / 2
        self.hbar_squared = self.hbar_matrix @ self.hbar_matrix
        self.hbar_operator = qml.dot(
            problem.coefficients / problem.pauli_bound, problem.operators
        )

    def _device(self, n_wires: int):
        seed = self.seed + self._device_counter
        self._device_counter += 1
        return qml.device(
            self.device_name, wires=n_wires, shots=self.shots, seed=seed
        )

    def _mapped_hamiltonian(self, data_wires: Sequence[int]):
        return qml.map_wires(
            self.hbar_operator,
            {source: target for source, target in enumerate(data_wires)},
        )

    def build_alpha_circuit(self, state: Array):
        """Build a circuit to estimate the state's first moment."""

        preparation = _state_preparation_unitary(state)
        data = list(range(self.problem.n_qubits))
        if self.epsilon == 0:
            mapped_hamiltonian = self._mapped_hamiltonian(data)
            device = self._device(self.problem.n_qubits)

            @qml.qnode(device)
            def circuit():
                qml.QubitUnitary(preparation, wires=data)
                return qml.expval(mapped_hamiltonian)

            return circuit

        test_wire = self.problem.n_qubits
        device = self._device(test_wire + 1)

        @qml.qnode(device)
        def circuit():
            qml.QubitUnitary(preparation, wires=data)
            qml.Hadamard(test_wire)
            qml.ctrl(qml.QubitUnitary, control=test_wire)(
                self.evolution_matrix, wires=data
            )
            qml.adjoint(qml.S)(test_wire)
            qml.Hadamard(test_wire)
            return qml.probs(wires=[test_wire])

        return circuit

    def build_second_moment_circuit(self, state: Array):
        """Build a circuit to estimate the state's second moment."""

        preparation = _state_preparation_unitary(state)
        data = list(range(self.problem.n_qubits))
        if self.epsilon == 0:
            device = self._device(self.problem.n_qubits)

            @qml.qnode(device)
            def circuit():
                qml.QubitUnitary(preparation, wires=data)
                return qml.expval(qml.Hermitian(self.hbar_squared, wires=data))

            return circuit

        test_wire = self.problem.n_qubits
        device = self._device(test_wire + 1)

        @qml.qnode(device)
        def circuit():
            qml.QubitUnitary(preparation, wires=data)
            qml.Hadamard(test_wire)
            qml.ctrl(qml.QubitUnitary, control=test_wire)(
                self.double_evolution_matrix, wires=data
            )
            qml.Hadamard(test_wire)
            return qml.probs(wires=[test_wire])

        return circuit

    def measure_moments(self, state: Array) -> Tuple[float, float]:
        """Execute the PDF's two moment-measurement circuits."""

        alpha_measurement = self.build_alpha_circuit(state)()
        second_measurement = self.build_second_moment_circuit(state)()
        if self.epsilon == 0:
            alpha = float(alpha_measurement)
            second_moment = float(second_measurement)
        else:
            alpha_probabilities = np.asarray(alpha_measurement, dtype=float)
            second_probabilities = np.asarray(second_measurement, dtype=float)
            imaginary_overlap = float(
                alpha_probabilities[0] - alpha_probabilities[1]
            )
            real_double_overlap = float(
                second_probabilities[0] - second_probabilities[1]
            )
            alpha = -imaginary_overlap / self.epsilon
            second_moment = (
                1.0 - real_double_overlap
            ) / (2.0 * self.epsilon**2)
        if second_moment < -self.moment_tolerance:
            raise FloatingPointError("the measured second moment is negative")
        return alpha, max(0.0, second_moment)

    def _build_diagonal_projection_circuit(self, state: Array):
        preparation = _state_preparation_unitary(state)
        data = list(range(self.problem.n_qubits))
        mapped_hamiltonian = self._mapped_hamiltonian(data)
        device = self._device(self.problem.n_qubits)

        @qml.qnode(device)
        def circuit():
            qml.QubitUnitary(preparation, wires=data)
            return qml.expval(mapped_hamiltonian)

        return circuit

    def build_transition_circuit(
        self,
        left_state: Array,
        right_state: Array,
        *,
        observable: str,
        component: str,
    ):
        """Build an interference circuit for an overlap or Hamiltonian element."""

        if observable not in {"overlap", "hamiltonian"}:
            raise ValueError("observable must be 'overlap' or 'hamiltonian'")
        if component not in {"real", "imag"}:
            raise ValueError("component must be 'real' or 'imag'")
        left_preparation = _state_preparation_unitary(left_state)
        right_preparation = _state_preparation_unitary(right_state)
        data = list(range(self.problem.n_qubits))
        test_wire = self.problem.n_qubits
        mapped_hamiltonian = self._mapped_hamiltonian(data)
        device = self._device(test_wire + 1)

        @qml.qnode(device)
        def circuit():
            qml.Hadamard(test_wire)
            qml.ctrl(qml.QubitUnitary, control=test_wire, control_values=[0])(
                left_preparation, wires=data
            )
            qml.ctrl(qml.QubitUnitary, control=test_wire, control_values=[1])(
                right_preparation, wires=data
            )
            test_observable = (
                qml.PauliX(test_wire)
                if component == "real"
                else qml.PauliY(test_wire)
            )
            measured = test_observable
            if observable == "hamiltonian":
                measured = measured @ mapped_hamiltonian
            return qml.expval(measured)

        return circuit

    def _project_subspace(self, states: Sequence[Array]) -> Tuple[Array, Array]:
        dimension = len(states)
        overlap = np.eye(dimension, dtype=complex)
        projected = np.zeros((dimension, dimension), dtype=complex)
        for index, state in enumerate(states):
            projected[index, index] = float(
                self._build_diagonal_projection_circuit(state)()
            )
        for row in range(dimension):
            for column in range(row):
                overlap_real = float(
                    self.build_transition_circuit(
                        states[row], states[column],
                        observable="overlap", component="real",
                    )()
                )
                overlap_imag = float(
                    self.build_transition_circuit(
                        states[row], states[column],
                        observable="overlap", component="imag",
                    )()
                )
                h_real = float(
                    self.build_transition_circuit(
                        states[row], states[column],
                        observable="hamiltonian", component="real",
                    )()
                )
                h_imag = float(
                    self.build_transition_circuit(
                        states[row], states[column],
                        observable="hamiltonian", component="imag",
                    )()
                )
                overlap[row, column] = overlap_real + 1j * overlap_imag
                projected[row, column] = h_real + 1j * h_imag
                overlap[column, row] = overlap[row, column].conj()
                projected[column, row] = projected[row, column].conj()
        return (
            (overlap + overlap.conj().T) / 2,
            (projected + projected.conj().T) / 2,
        )

    def _overlap_diagnostics(self, overlap: Array) -> Tuple[int, float, float]:
        eigenvalues = np.linalg.eigvalsh(overlap)
        largest = float(eigenvalues[-1])
        threshold = self.overlap_tolerance * max(1.0, largest)
        retained = eigenvalues > threshold
        rank = int(np.count_nonzero(retained))
        condition = (
            float(eigenvalues[retained][-1] / eigenvalues[retained][0])
            if rank
            else np.inf
        )
        error = float(np.max(np.abs(overlap - np.eye(len(overlap)))))
        return rank, condition, error

    def run(self) -> OCQKUDResult:
        """Execute Steps 1-13 of the PDF in their stated order."""

        coefficients = [np.array([1.0 + 0.0j])]
        states: list[Array] = []
        alpha_values = []
        second_moments = []
        beta_values = [0.0]
        history = []

        for n in range(self.max_dimension):
            state, raw_norm = _clenshaw_state(
                self.action_matrix, self.problem.hf_state, coefficients[-1]
            )
            if n > 0 and abs(raw_norm - 1.0) > self.state_norm_tolerance:
                coefficients.pop()
                break
            coefficients[-1] = coefficients[-1] / raw_norm
            alpha, second_moment = self.measure_moments(state)
            beta_n = beta_values[n]
            beta_squared = second_moment - alpha**2 - beta_n**2
            moment_scale = max(1.0, second_moment, alpha**2, beta_n**2)
            if beta_squared < -self.moment_tolerance * moment_scale:
                raise FloatingPointError(
                    f"measured beta_{n + 1}^2={beta_squared:.3e} is inconsistent"
                )
            beta_next = float(np.sqrt(max(0.0, beta_squared)))
            states.append(state)
            alpha_values.append(alpha)
            second_moments.append(second_moment)
            history.append(
                {
                    "K": n + 1,
                    "alpha": alpha,
                    "second_moment": second_moment,
                    "beta_next": beta_next,
                    "state_norm_before_normalization": raw_norm,
                }
            )
            if n + 1 == self.max_dimension or beta_next <= self.breakdown_tolerance:
                break
            previous = coefficients[-2] if n > 0 else None
            coefficients.append(
                _next_chebyshev_coefficients(
                    previous, coefficients[-1], alpha, beta_n, beta_next
                )
            )
            beta_values.append(beta_next)

        alpha_array = np.asarray(alpha_values, dtype=float)
        beta_array = np.asarray(beta_values[: len(alpha_array)], dtype=float)
        overlap, projected = self._project_subspace(states)
        physical_dimensionless, ritz_coefficients = np.linalg.eigh(projected)
        for index, row in enumerate(history):
            prefix_overlap = overlap[: index + 1, : index + 1]
            prefix_projected = projected[: index + 1, : index + 1]
            prefix_energy = float(np.linalg.eigvalsh(prefix_projected)[0])
            rank, condition, orthogonality_error = self._overlap_diagnostics(
                prefix_overlap
            )
            row["energy_dimensionless"] = prefix_energy
            row["energy_hartree"] = self.problem.pauli_bound * prefix_energy
            row["overlap_rank"] = rank
            row["overlap_condition"] = condition
            row["orthogonality_error"] = orthogonality_error

        return OCQKUDResult(
            epsilon=self.epsilon,
            scale=self.problem.pauli_bound,
            alpha=alpha_array,
            second_moments=np.asarray(second_moments, dtype=float),
            beta=beta_array,
            chebyshev_coefficients=coefficients,
            state_vectors=states,
            overlap=overlap,
            projected_hamiltonian=projected,
            physical_energies=self.problem.pauli_bound * physical_dimensionless,
            ritz_coefficients=ritz_coefficients,
            history=history,
        )


__all__ = [
    "MolecularProblem",
    "OCQKUDResult",
    "QuantumOCQKUD",
    "build_molecular_problem",
]
