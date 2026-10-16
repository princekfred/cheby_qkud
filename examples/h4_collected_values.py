"""Generate the landscape H4 collected-expectation document and its data."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from ocqkudsin import QuantumOCQKUDSIN
from examples.h4 import ARCHIVE_FILE, load_archived_problem


EPSILONS = (0.5, 0.8, 1.2)
N_STATES = 6
PRECISION = 90
DOCS = ROOT / "docs"
STEM = "h4_collected_values_cqkud_landscape"


def _unshifted_moment(real: Decimal, imag: Decimal, k: int) -> tuple[Decimal, Decimal]:
    """Return <H V^k> from <H (iV)^k>, where V=exp(-i epsilon Hbar)."""

    return (
        (real, imag),
        (imag, -real),
        (-real, -imag),
        (-imag, real),
    )[k % 4]


def _convert_row(row: dict) -> dict:
    """Express one result with the reference PDF's unshifted V convention."""

    n = int(row["n"])
    multiplier = Decimal(4) ** n
    terms = []
    with localcontext() as context:
        context.prec = 140
        denominator = multiplier * Decimal(row["denominator"])
        numerator = Decimal(0)
        expanded_sum = Decimal(0)
        for raw in row["terms"]:
            k = int(raw["k"])
            coefficient = multiplier * Decimal(raw["coefficient"])
            shifted_real = Decimal(raw["moment_real"])
            shifted_imag = Decimal(raw["moment_imag"])
            moment_real, moment_imag = _unshifted_moment(
                shifted_real, shifted_imag, k
            )
            if k == 0:
                coefficient_real = coefficient
                coefficient_imag = Decimal(0)
                bracket_real = moment_real
                bracket_imag = Decimal(0)
            elif k % 2 == 0:
                coefficient_real = coefficient * (-1 if k // 2 % 2 else 1)
                coefficient_imag = Decimal(0)
                bracket_real = 2 * moment_real
                bracket_imag = Decimal(0)
            else:
                coefficient_real = Decimal(0)
                coefficient_imag = coefficient * (
                    -1 if (k + 1) // 2 % 2 else 1
                )
                bracket_real = Decimal(0)
                bracket_imag = -2 * moment_imag
            numerator += (
                coefficient_real * bracket_real
                - coefficient_imag * bracket_imag
            )
            expanded_real = coefficient_real / denominator
            expanded_imag = coefficient_imag / denominator
            expanded_sum += (
                expanded_real * bracket_real
                - expanded_imag * bracket_imag
            )
            terms.append(
                {
                    "k": k,
                    "coefficient_real": str(expanded_real),
                    "coefficient_imag": str(expanded_imag),
                    "numerator_coefficient_real": str(coefficient_real),
                    "numerator_coefficient_imag": str(coefficient_imag),
                    "bracket_real_hartree": str(bracket_real),
                    "bracket_imag_hartree": str(bracket_imag),
                    "moment_real_hartree": str(moment_real),
                    "moment_imag_hartree": str(moment_imag),
                }
            )
        expected = Decimal(row["expectation_H_hartree"])
        difference = numerator / denominator - expected
        if abs(difference) > Decimal("1e-65"):
            raise ArithmeticError(
                f"unshifted expression does not agree at n={n}: {difference}"
            )
        if abs(expanded_sum - expected) > Decimal("1e-65"):
            raise ArithmeticError(
                f"expanded expression does not agree at n={n}: "
                f"{expanded_sum - expected}"
            )
    return {
        "n": n,
        "denominator": str(denominator),
        "expectation_H_hartree": row["expectation_H_hartree"],
        "terms": terms,
    }


def _tex_decimal(value: Decimal, significant: int = 17) -> str:
    formatted = format(abs(value), f".{significant}g")
    if "e" in formatted.lower():
        mantissa, exponent = formatted.lower().split("e")
        return mantissa + r"\times 10^{" + str(int(exponent)) + "}"
    return formatted


def _bracket_tex(k: int) -> str:
    if k == 0:
        return r"\langle\Phi_0|H|\Phi_0\rangle"
    left = (
        r"\langle\Phi_0|H(V^\dagger)^{"
        + str(k)
        + r"}|\Phi_0\rangle"
    )
    right = (
        r"\langle\Phi_0|HV^{"
        + str(k)
        + r"}|\Phi_0\rangle"
    )
    operator = "+" if k % 2 == 0 else "-"
    return r"\bigl(" + left + operator + right + r"\bigr)"


def _term_tex(term: dict) -> tuple[str, str]:
    k = term["k"]
    coefficient = Decimal(
        term["coefficient_imag"] if k % 2 else term["coefficient_real"]
    )
    bracket = Decimal(
        term["bracket_imag_hartree"] if k % 2 else term["bracket_real_hartree"]
    )
    sign = "-" if coefficient < 0 else "+"
    coefficient_tex = _tex_decimal(coefficient, 16)
    if k % 2:
        coefficient_tex += r"\,i"
    bracket_tex = _bracket_tex(k)
    bracket_value = ("-" if bracket < 0 else "") + _tex_decimal(bracket, 20)
    if k % 2:
        bracket_value += r"\,i"
    underbrace = (
        r"\underbrace{" + bracket_tex + r"}_{{\scriptstyle\approx "
        + bracket_value + r"}}"
    )
    return sign, coefficient_tex + r"\," + underbrace


def _row_tex(row: dict, epsilon: float) -> str:
    n = row["n"]
    heading = (
        r"\begin{minipage}{\linewidth}" + "\n"
        + r"\textbf{Iteration $n=" + str(n)
        + r"$, $\epsilon=" + f"{epsilon:g}" + r"$}" + "\n\n"
        + r"\textbf{4. Collected final expression}" + "\n"
    )
    if n == 0:
        term = _term_tex(row["terms"][0])[1]
        if term.startswith(r"1\,"):
            term = term[3:]
        formula = (
            r"\[\begin{aligned}"
            + r"\langle q_0|H|q_0\rangle &\simeq "
            + term
            + r"\\[1.2em]"
            + r"&\simeq "
            + format(Decimal(row["expectation_H_hartree"]), ".14f")
            + r"\ \mathrm{Hartree}."
            + r"\end{aligned}\]"
        )
    else:
        pieces = [_term_tex(term) for term in row["terms"]]
        lines = []
        for start in range(0, len(pieces), 2):
            line = ""
            for index in range(start, min(start + 2, len(pieces))):
                sign, body = pieces[index]
                prefix = "" if index == 0 and sign == "+" else sign
                if index == start:
                    line += "&" + prefix + body
                else:
                    line += r"\qquad " + prefix + body
            lines.append(line)
        formula = (
            r"\[\begin{aligned}"
            + r"\langle q_{"
            + str(n)
            + r"}|H|q_{"
            + str(n)
            + r"}\rangle &\simeq \begin{aligned}[t]"
            + (r"\\[1.45em]" + "\n").join(lines)
            + r"\end{aligned}\\[1.4em]"
            + r"&\simeq "
            + format(Decimal(row["expectation_H_hartree"]), ".14f")
            + r"\ \mathrm{Hartree}."
            + r"\end{aligned}\]"
        )
    return (
        heading
        + r"\begingroup\small"
        + "\n"
        + formula
        + "\n"
        + r"\endgroup"
        + "\n"
        + r"\end{minipage}"
        + "\n\n"
        + r"\vspace{1em}"
        + "\n"
    )


def _document_tex(results: list[dict]) -> str:
    preamble = r"""\documentclass[10pt,letterpaper,landscape]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=0.52in}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.35em}
\pagestyle{plain}
\begin{document}
\begin{center}
{\Large Square H4: CQKUD collected expectations with numerical values}\\[0.3em]
{\large First six states at $\epsilon=0.5,\,0.8,\,1.2$}
\end{center}
\small
Only the collected final expression is shown for each state. The number beneath
each expectation or paired bracket is its value in Hartree; full-precision
coefficients and complex moments are in the accompanying JSON.
Here $H=\Lambda\bar H$ with $\Lambda=\texttt{__SCALE__}$ Hartree,
$V=e^{-i\epsilon\bar H}$, and
$X_\epsilon=\sin(\epsilon\bar H)=(V^\dagger-V)/(2i)$.
The solver uses $U_\epsilon=iV$ internally; this display converts
to the unshifted notation of the reference PDF. The common denominator
$4^n\prod_{j=1}^{n}\beta_j^2$ is divided into every displayed coefficient;
the JSON also retains the original numerator coefficients and denominator.
Printed values are rounded; final values use
unrounded \texttt{Decimal} spectral sums for the saved float64 Hamiltonian.
\normalsize

"""
    preamble = preamble.replace(
        "__SCALE__", _tex_decimal(Decimal(results[0]["scale_hartree"]), 17)
    )
    chunks = [preamble]
    for index, result in enumerate(results):
        if index:
            chunks.append(r"\clearpage" + "\n")
        chunks.append(r"\section*{$\epsilon=" + result["epsilon"] + r"$}" + "\n")
        for row in result["states"]:
            chunks.append(_row_tex(row, float(result["epsilon"])))
    chunks.append(r"\end{document}" + "\n")
    return "".join(chunks)


def main() -> None:
    problem = load_archived_problem()
    with np.load(ARCHIVE_FILE, allow_pickle=False) as snapshot:
        sector_indices = np.asarray(snapshot["sector_indices"], dtype=int)
    results = []
    for epsilon in EPSILONS:
        raw = QuantumOCQKUDSIN(
            problem, epsilon=epsilon, max_dimension=N_STATES
        ).collected_expressions(
            n_states=N_STATES,
            sector_indices=sector_indices,
            precision=PRECISION,
        )
        results.append(
            {
                "epsilon": raw["epsilon"],
                "scale_hartree": raw["scale_hartree"],
                "states": [_convert_row(row) for row in raw["states"]],
            }
        )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / f"{STEM}.tex").write_text(_document_tex(results), encoding="utf-8")
    (DOCS / f"{STEM}.json").write_text(
        json.dumps(
            {
                "convention": "V_epsilon=exp(-i epsilon Hbar); X_epsilon=sin(epsilon Hbar)",
                "source": str(ARCHIVE_FILE.relative_to(ROOT)),
                "precision_decimal_digits": PRECISION,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
