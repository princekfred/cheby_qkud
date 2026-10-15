"""Exact circuit realization of the algorithm specified in ``ocqkud.pdf``.

This variant uses X_epsilon = sin(epsilon * Hbar), with X_0 = 0. Define the
phase-shifted unitary U_epsilon = i * exp(-i * epsilon * Hbar); then
X_epsilon = (U_epsilon.conj().T + U_epsilon) / 2. The phase shift is needed
because the unshifted time evolution in the PDF gives cosine in this sum.

The classical part stores only the Chebyshev coefficient lists and performs
the PDF's coefficient recurrence. At each iteration, the current polynomial
state is evaluated by the stated Clenshaw recurrence and normalized. A
PennyLane state-preparation circuit then prepares that state for the moment
measurements. Controlled exact time-evolution circuits measure the two
moments using the phase-shifted unitary. Their formulas are adjusted for the
missing 1/epsilon factor in the working operator. For very small nonzero
epsilon, direct Hermitian-observable circuits avoid cancellation in the
Hadamard-test probabilities.
The Chebyshev argument is rescaled by |epsilon| in that small-epsilon regime
so its coefficients remain numerically well conditioned.
At epsilon zero, both moments vanish and the Krylov space has one state.

The projected Hamiltonian and overlap matrix are also measured by circuits.
For exact measurements, Eq. (37) of the PDF gives physical Ritz values by
diagonalizing the projected Hamiltonian M. With finite shots, the measured
overlap S is included in a generalized eigenproblem because the generated
states are no longer exactly orthogonal.

Exact generic state preparation and exact dense time-evolution gates scale
exponentially. They make this module a small-system adaptation of the PDF,
not a scalable fault-tolerant construction. Replacing these
gates by hardware decompositions introduces an approximation not specified
by the PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext
from functools import lru_cache
from math import comb, factorial
from typing import Optional, Sequence, Tuple

import numpy as np
import pennylane as qml


Array = np.ndarray


@dataclass(frozen=True)
class MolecularProblem:
    """Molecular qubit Hamiltonian and Hartree-Fock reference data."""

    hamiltonian: object
    coefficients: Array
    operators: tuple
    n_qubits: int
    hf_bits: Array
    hf_state: Array
    pauli_bound: float


@dataclass
class OCQKUDResult:
    """Outputs of :meth:`QuantumOCQKUD.run`."""

    epsilon: float
    scale: float
    alpha: Array
    second_moments: Array
    beta: Array
    chebyshev_coefficients: list[Array]
    chebyshev_operator_scale: float
    chebyshev_state_vectors: list[Array]
    state_vectors: list[Array]
    effective_matrix: Array
    effective_eigenvalues: Array
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
    coefficients = np.real_if_close(np.asarray(coefficients, dtype=complex)).astype(float)
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


def balanced_spin_sector(
    n_qubits: int, n_alpha: int, n_beta: int
) -> Array:
    """Return computational-basis indices in a fixed spin sector."""

    indices = []
    for index in range(2**n_qubits):
        bits = [int(bit) for bit in f"{index:0{n_qubits}b}"]
        if sum(bits[::2]) == n_alpha and sum(bits[1::2]) == n_beta:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def exact_sector_energy(
    problem: MolecularProblem, n_alpha: int, n_beta: int
) -> float:
    """Return an exact-sector benchmark that is not used by OCQKUD."""

    matrix = np.asarray(
        qml.matrix(problem.hamiltonian, wire_order=range(problem.n_qubits)),
        dtype=complex,
    )
    sector = balanced_spin_sector(problem.n_qubits, n_alpha, n_beta)
    return float(np.linalg.eigvalsh(matrix[np.ix_(sector, sector)])[0])


def _state_preparation_unitary(state: Array) -> Array:
    """Return a unitary whose first column is the normalized ``state``."""

    target = np.asarray(state, dtype=complex)
    state_norm = np.linalg.norm(target)
    if not np.isfinite(state_norm) or state_norm == 0:
        raise ValueError("state must be finite and nonzero")
    target = target / state_norm
    phase = 1.0 + 0.0j
    if abs(target[0]) > 1e-15:
        phase = target[0] / abs(target[0])
        target = target / phase
    origin = np.zeros_like(target)
    origin[0] = 1.0
    difference = origin - target
    norm = np.linalg.norm(difference)
    if norm < 1e-14:
        return phase * np.eye(len(target), dtype=complex)
    direction = difference / norm
    unitary = np.eye(len(target), dtype=complex) - 2.0 * np.outer(
        direction, direction.conj()
    )
    if not np.allclose(unitary.conj().T @ unitary, np.eye(len(target)), atol=1e-10):
        raise FloatingPointError("state-preparation matrix is not unitary")
    return phase * unitary


def _next_chebyshev_coefficients(
    previous: Optional[Array],
    current: Array,
    alpha: float,
    beta: float,
    beta_next: float,
) -> Array:
    """Implement Eqs. (28)-(33), including the T0 endpoint rule."""

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
    """Evaluate Eqs. (15)-(18) without storing all Chebyshev vectors."""

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


def _chebyshev_state_vectors(
    action_matrix: Array, reference_state: Array, max_degree: int
) -> list[Array]:
    """Return every ``T_m(action_matrix) @ reference_state`` through max_degree."""

    if max_degree < 0:
        return []
    vectors = [np.asarray(reference_state, dtype=complex).copy()]
    if max_degree == 0:
        return vectors
    vectors.append(action_matrix @ vectors[0])
    for _degree in range(2, max_degree + 1):
        vectors.append(2.0 * action_matrix @ vectors[-1] - vectors[-2])
    return vectors


@lru_cache(maxsize=32)
def _decimal_pi(precision: int) -> Decimal:
    """Compute pi with enough digits for Decimal argument reduction."""

    with localcontext() as context:
        context.prec = precision + 8
        series = Decimal(0)
        for k in range((precision + 8) // 14 + 2):
            numerator = Decimal(factorial(6 * k) * (13591409 + 545140134 * k))
            denominator = Decimal(
                factorial(3 * k) * factorial(k) ** 3 * 640320 ** (3 * k)
            )
            series += (-1 if k % 2 else 1) * numerator / denominator
        result = 426880 * Decimal(10005).sqrt() / series
        context.prec = precision
        return +result


def _decimal_sincos(value: Decimal) -> Tuple[Decimal, Decimal]:
    """Evaluate sine and cosine with Decimal argument reduction."""

    precision = getcontext().prec
    reduction_precision = precision + max(0, value.adjusted() + 1) + 10
    with localcontext() as context:
        context.prec = reduction_precision
        pi = _decimal_pi(reduction_precision)
        two_pi = 2 * pi
        reduced = value % two_pi
        if reduced > pi:
            reduced -= two_pi
        elif reduced < -pi:
            reduced += two_pi

        context.prec = precision + 8
        x = +reduced
        square = x * x
        sine_term = x
        cosine_term = Decimal(1)
        sine = sine_term
        cosine = cosine_term
        cutoff = Decimal(1).scaleb(-precision - 6)
        for n in range(1, max(300, 2 * precision)):
            sine_term *= -square / Decimal((2 * n) * (2 * n + 1))
            cosine_term *= -square / Decimal((2 * n - 1) * (2 * n))
            sine += sine_term
            cosine += cosine_term
            if abs(sine_term) < cutoff and abs(cosine_term) < cutoff:
                context.prec = precision
                return +sine, +cosine
    raise ArithmeticError("the reference sine/cosine series did not converge")


def _monic_polynomials(
    alpha: Sequence[Decimal], beta: Sequence[Decimal]
) -> list[list[Decimal]]:
    """Construct monic Lanczos polynomials in ascending powers of X."""

    output = [[Decimal(1)]]
    for n in range(len(alpha) - 1):
        current = output[-1]
        following = [Decimal(0) for _ in range(n + 2)]
        for power, value in enumerate(current):
            following[power + 1] += value
            following[power] -= alpha[n] * value
        if n:
            for power, value in enumerate(output[-2]):
                following[power] -= beta[n] ** 2 * value
        output.append(following)
    return output


def _collected_unitary_coefficients(polynomial: Sequence[Decimal]) -> list[Decimal]:
    """Expand P_n(X)^2 for X=(U + U†)/2 without a 4**n multiplier."""

    n = len(polynomial) - 1
    squared = [Decimal(0) for _ in range(2 * n + 1)]
    for left_power, left in enumerate(polynomial):
        for right_power, right in enumerate(polynomial):
            squared[left_power + right_power] += left * right
    return [
        sum(
            (
                value * Decimal(comb(power, (power - k) // 2))
                / Decimal(2) ** power
                for power, value in enumerate(squared)
                if power >= k and (power - k) % 2 == 0
            ),
            Decimal(0),
        )
        for k in range(2 * n + 1)
    ]


class QuantumOCQKUD:
    """Exact small-system circuit implementation of ``ocqkud.pdf``."""

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

        # X_epsilon = sin(epsilon * Hbar). A global phase on the PDF's
        # time-evolution unitary makes its Hermitian part equal to X_epsilon.
        sine_matrix = (
            eigenvectors * np.sin(epsilon * eigenvalues)
        ) @ eigenvectors.conj().T
        shifted_phases = 1j * np.exp(-1j * epsilon * eigenvalues)
        self.evolution_matrix = (
            eigenvectors * shifted_phases
        ) @ eigenvectors.conj().T
        self.double_evolution_matrix = (
            eigenvectors * shifted_phases**2
        ) @ eigenvectors.conj().T
        unitary_action = (
            self.evolution_matrix.conj().T + self.evolution_matrix
        ) / 2
        if not np.allclose(unitary_action, sine_matrix, atol=1e-10):
            raise FloatingPointError("the two definitions of X_epsilon disagree")
        self.action_matrix = (
            sine_matrix + sine_matrix.conj().T
        ) / 2
        self.action_squared = self.action_matrix @ self.action_matrix
        self._use_direct_moments = 0 < abs(epsilon) < 1e-3
        self.chebyshev_operator_scale = (
            abs(epsilon) if self._use_direct_moments else 1.0
        )
        self.chebyshev_matrix = (
            self.action_matrix / self.chebyshev_operator_scale
        )
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
        """Measure <state|X_epsilon|state>."""

        preparation = _state_preparation_unitary(state)
        data = list(range(self.problem.n_qubits))
        if self._use_direct_moments:
            device = self._device(self.problem.n_qubits)

            @qml.qnode(device)
            def circuit():
                qml.QubitUnitary(preparation, wires=data)
                return qml.expval(qml.Hermitian(self.action_matrix, wires=data))

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
            qml.Hadamard(test_wire)
            return qml.probs(wires=[test_wire])

        return circuit

    def build_second_moment_circuit(self, state: Array):
        """Measure <state|X_epsilon**2|state>."""

        preparation = _state_preparation_unitary(state)
        data = list(range(self.problem.n_qubits))
        if self._use_direct_moments:
            device = self._device(self.problem.n_qubits)

            @qml.qnode(device)
            def circuit():
                qml.QubitUnitary(preparation, wires=data)
                return qml.expval(qml.Hermitian(self.action_squared, wires=data))

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
        """Execute the two moment-measurement circuits."""

        if self.epsilon == 0:
            return 0.0, 0.0
        alpha_measurement = self.build_alpha_circuit(state)()
        second_measurement = self.build_second_moment_circuit(state)()
        if self._use_direct_moments:
            alpha = float(alpha_measurement)
            second_moment = float(second_measurement)
        else:
            alpha_probabilities = np.asarray(alpha_measurement, dtype=float)
            second_probabilities = np.asarray(second_measurement, dtype=float)
            alpha = float(alpha_probabilities[0] - alpha_probabilities[1])
            real_double_overlap = float(
                second_probabilities[0] - second_probabilities[1]
            )
            second_moment = (1.0 + real_double_overlap) / 2.0
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
        """Build an interference circuit for Eq. (35) or Eq. (36)."""

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

    def _overlap_threshold(self, overlap: Array, largest: float) -> float:
        threshold = self.overlap_tolerance * max(1.0, largest)
        if self.shots is not None:
            threshold = max(
                threshold, 5.0 * np.sqrt(len(overlap) / self.shots)
            )
        return threshold

    def _overlap_diagnostics(self, overlap: Array) -> Tuple[int, float, float]:
        eigenvalues = np.linalg.eigvalsh(overlap)
        largest = float(eigenvalues[-1])
        threshold = self._overlap_threshold(overlap, largest)
        retained = eigenvalues > threshold
        rank = int(np.count_nonzero(retained))
        condition = (
            float(eigenvalues[retained][-1] / eigenvalues[retained][0])
            if rank
            else np.inf
        )
        error = float(np.max(np.abs(overlap - np.eye(len(overlap)))))
        return rank, condition, error

    def _diagonalize_projection(
        self, overlap: Array, projected: Array
    ) -> Tuple[Array, Array]:
        """Account for nonorthogonal states when measurements use finite shots."""

        if self.shots is None:
            return np.linalg.eigh(projected)
        overlap_values, overlap_vectors = np.linalg.eigh(overlap)
        threshold = self._overlap_threshold(overlap, float(overlap_values[-1]))
        retained = overlap_values > threshold
        if not np.any(retained):
            raise FloatingPointError("the measured overlap matrix has no positive rank")
        orthonormalizer = (
            overlap_vectors[:, retained] / np.sqrt(overlap_values[retained])
        )
        reduced = orthonormalizer.conj().T @ projected @ orthonormalizer
        reduced = (reduced + reduced.conj().T) / 2
        energies, vectors = np.linalg.eigh(reduced)
        return energies, orthonormalizer @ vectors

    def run(self) -> OCQKUDResult:
        """Execute the PDF's Krylov steps with the unscaled sine operator."""

        coefficients = [np.array([1.0 + 0.0j])]
        states: list[Array] = []
        alpha_values = []
        second_moments = []
        beta_values = [0.0]
        history = []

        for n in range(self.max_dimension):
            state, raw_norm = _clenshaw_state(
                self.chebyshev_matrix, self.problem.hf_state, coefficients[-1]
            )
            if (
                n > 0
                and self.shots is None
                and abs(raw_norm - 1.0) > self.state_norm_tolerance
            ):
                coefficients.pop()
                break
            coefficients[-1] = coefficients[-1] / raw_norm
            alpha, second_moment = self.measure_moments(state)
            beta_n = beta_values[n]
            beta_squared = second_moment - alpha**2 - beta_n**2
            moment_scale = max(
                self.chebyshev_operator_scale**2,
                second_moment,
                alpha**2,
                beta_n**2,
            )
            moment_error = self.moment_tolerance * moment_scale
            if self.shots is not None:
                shot_scale = (
                    self.chebyshev_operator_scale**2
                    if self._use_direct_moments else 1.0
                )
                moment_error += 5.0 * shot_scale / np.sqrt(self.shots)
            if beta_squared < -moment_error:
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
            if (
                n + 1 == self.max_dimension
                or beta_next
                <= self.breakdown_tolerance * self.chebyshev_operator_scale
            ):
                break
            previous = coefficients[-2] if n > 0 else None
            coefficients.append(
                _next_chebyshev_coefficients(
                    previous,
                    coefficients[-1],
                    alpha / self.chebyshev_operator_scale,
                    beta_n / self.chebyshev_operator_scale,
                    beta_next / self.chebyshev_operator_scale,
                )
            )
            beta_values.append(beta_next)

        alpha_array = np.asarray(alpha_values, dtype=float)
        beta_array = np.asarray(beta_values[: len(alpha_array)], dtype=float)
        chebyshev_states = _chebyshev_state_vectors(
            self.chebyshev_matrix, self.problem.hf_state, len(states) - 1
        )
        effective = np.diag(alpha_array)
        for index in range(1, len(alpha_array)):
            effective[index - 1, index] = beta_array[index]
            effective[index, index - 1] = beta_array[index]
        effective_eigenvalues = np.linalg.eigvalsh(effective)

        overlap, projected = self._project_subspace(states)
        physical_dimensionless, ritz_coefficients = self._diagonalize_projection(
            overlap, projected
        )
        for index, row in enumerate(history):
            prefix_overlap = overlap[: index + 1, : index + 1]
            prefix_projected = projected[: index + 1, : index + 1]
            prefix_energy = float(
                self._diagonalize_projection(
                    prefix_overlap, prefix_projected
                )[0][0]
            )
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
            chebyshev_operator_scale=self.chebyshev_operator_scale,
            chebyshev_state_vectors=chebyshev_states,
            state_vectors=states,
            effective_matrix=effective,
            effective_eigenvalues=effective_eigenvalues,
            overlap=overlap,
            projected_hamiltonian=projected,
            physical_energies=self.problem.pauli_bound * physical_dimensionless,
            ritz_coefficients=ritz_coefficients,
            history=history,
        )

    def collected_expressions(
        self,
        *,
        n_states: Optional[int] = None,
        sector_indices: Optional[Sequence[int]] = None,
        precision: int = 90,
    ) -> dict:
        """Collect physical-H moments for X=(U+U†)/2, U=i exp(-i epsilon Hbar).

        The numerator directly expands each normalized state's monic Lanczos
        polynomial in U powers. Its denominator is product(beta_j**2), with
        no common 4**n multiplier. Decimal spectral sums provide reference
        moments from the floating-point Hamiltonian supplied to this solver.
        Small epsilon uses extra Decimal guard digits for cancellation in the
        collected numerator.
        """

        explicit_count = n_states is not None
        count = self.max_dimension if n_states is None else int(n_states)
        if count < 1:
            raise ValueError("n_states must be positive")
        if precision < 50:
            raise ValueError("precision must be at least 50 decimal digits")

        full_dimension = len(self.problem.hf_state)
        if sector_indices is None:
            indices = np.arange(full_dimension)
        else:
            raw_indices = np.asarray(sector_indices)
            if raw_indices.ndim != 1 or not np.issubdtype(
                raw_indices.dtype, np.integer
            ):
                raise ValueError("sector_indices must be a one-dimensional integer list")
            indices = raw_indices.astype(int)
            if (
                len(indices) == 0
                or np.any(indices < 0)
                or np.any(indices >= full_dimension)
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError("sector_indices must contain distinct valid indices")
            omitted = np.setdiff1d(np.arange(full_dimension), indices)
            if omitted.size and (
                np.linalg.norm(self.problem.hf_state[omitted]) > 1e-10
                or not np.allclose(
                    self.hbar_matrix[np.ix_(omitted, indices)], 0, atol=1e-10
                )
            ):
                raise ValueError("the selected sector does not contain an invariant reference state")

        hbar = self.hbar_matrix[np.ix_(indices, indices)]
        reference = np.asarray(self.problem.hf_state[indices], dtype=complex)
        eigenvalues, eigenvectors = np.linalg.eigh(hbar)
        weights_float = np.abs(eigenvectors.conj().T @ reference) ** 2

        with localcontext() as context:
            epsilon_digits = (
                max(0, -Decimal(str(abs(self.epsilon))).adjusted())
                if self.epsilon else 0
            )
            context.prec = precision + 2 * (count - 1) * epsilon_digits + 20
            zero = Decimal(0)
            epsilon = Decimal(str(self.epsilon))
            scale = Decimal.from_float(float(self.problem.pauli_bound))
            energies = [Decimal.from_float(float(value)) for value in eigenvalues]
            weights = [Decimal.from_float(float(value)) for value in weights_float]
            total_weight = sum(weights, zero)
            if total_weight <= 0:
                raise ValueError("the reference state has no spectral weight")
            weights = [weight / total_weight for weight in weights]
            action = [_decimal_sincos(epsilon * energy)[0] for energy in energies]

            alpha: list[Decimal] = []
            beta: list[Decimal] = [zero]
            states: list[list[Decimal]] = []
            current = [weight.sqrt() for weight in weights]
            previous = [zero for _ in weights]
            for n in range(count):
                states.append(current)
                alpha_n = sum(
                    (amplitude * amplitude * value
                     for amplitude, value in zip(current, action)),
                    zero,
                )
                residual = [
                    (value - alpha_n) * amplitude - beta[n] * old
                    for value, amplitude, old in zip(action, current, previous)
                ]
                beta_next = sum((value * value for value in residual), zero).sqrt()
                alpha.append(alpha_n)
                beta.append(beta_next)
                if n < count - 1:
                    if beta_next <= Decimal("1e-30"):
                        if explicit_count:
                            raise ArithmeticError(
                                f"the recurrence ended before q{count - 1}"
                            )
                        break
                    previous, current = current, [
                        value / beta_next for value in residual
                    ]

            moments: list[tuple[Decimal, Decimal]] = []
            for k in range(2 * (len(states) - 1) + 1):
                real = zero
                imag = zero
                for energy, weight in zip(energies, weights):
                    sine, cosine = _decimal_sincos(Decimal(k) * epsilon * energy)
                    component_real, component_imag = (
                        (cosine, -sine),
                        (sine, cosine),
                        (-cosine, sine),
                        (-sine, -cosine),
                    )[k % 4]
                    factor = scale * weight * energy
                    real += factor * component_real
                    imag += factor * component_imag
                moments.append((real, imag))

            rows = []
            beta_product_squared = Decimal(1)
            tolerance = Decimal(1).scaleb(-(precision - 35))
            for n, (polynomial, state) in enumerate(
                zip(_monic_polynomials(alpha, beta), states)
            ):
                if n:
                    beta_product_squared *= beta[n] ** 2
                terms = []
                numerator = zero
                for k, coefficient in enumerate(
                    _collected_unitary_coefficients(polynomial)
                ):
                    moment_real, moment_imag = moments[k]
                    bracket = moment_real if k == 0 else 2 * moment_real
                    numerator += coefficient * bracket
                    terms.append(
                        {
                            "k": k,
                            "coefficient": str(coefficient),
                            "moment_real": str(moment_real),
                            "moment_imag": str(moment_imag),
                            "bracket_hartree": str(bracket),
                        }
                    )
                grouped_value = numerator / beta_product_squared
                direct_value = scale * sum(
                    (energy * amplitude * amplitude
                     for energy, amplitude in zip(energies, state)),
                    zero,
                )
                if abs(grouped_value - direct_value) > tolerance * max(
                    Decimal(1), abs(direct_value)
                ):
                    raise ArithmeticError(f"collected expression disagrees at q{n}")
                rows.append(
                    {
                        "n": n,
                        "denominator": str(beta_product_squared),
                        "expectation_H_hartree": str(grouped_value),
                        "terms": terms,
                    }
                )
            return {
                "epsilon": str(epsilon),
                "scale_hartree": str(scale),
                "alpha": [str(value) for value in alpha],
                "beta": [str(value) for value in beta],
                "states": rows,
            }

    def draw_alpha_circuit(self, state: Array, level: str = "top") -> str:
        """Return a text diagram of the alpha measurement circuit."""

        return qml.draw(self.build_alpha_circuit(state), level=level)()

    def draw_second_moment_circuit(self, state: Array, level: str = "top") -> str:
        """Return a text diagram of the second-moment circuit."""

        return qml.draw(self.build_second_moment_circuit(state), level=level)()

    def draw_transition_circuit(
        self,
        left_state: Array,
        right_state: Array,
        *,
        observable: str = "hamiltonian",
        component: str = "real",
        level: str = "top",
    ) -> str:
        """Return a text diagram of an S or Hbar transition circuit."""

        circuit = self.build_transition_circuit(
            left_state, right_state,
            observable=observable, component=component,
        )
        return qml.draw(circuit, level=level)()


__all__ = [
    "MolecularProblem",
    "OCQKUDResult",
    "QuantumOCQKUD",
    "balanced_spin_sector",
    "build_molecular_problem",
    "exact_sector_energy",
]
