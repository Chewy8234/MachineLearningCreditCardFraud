"""Publication-ready LaTeX table fragments (booktabs), generated without templates.

Each fragment is a complete ``table`` environment that can be ``\\input`` into a
paper (requires ``\\usepackage{booktabs}``). Bolding marks the best value per
group *as computed*, never an assumed winner; reference rows (e.g. the
centralized, non-federated condition) can be excluded from bolding.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

_LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                  "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


def escape(text: object) -> str:
    s = str(text)
    out = "".join(_LATEX_ESCAPES.get(c, c) for c in s)
    return out.replace("α", r"$\alpha$").replace("λ", r"$\lambda$").replace("±", r"$\pm$")


def fmt_mean_std(mean: float, std: float | None, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_num(v: object, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer() and abs(v) >= 100):
        return f"{int(v):,}".replace(",", "{,}")
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return escape(v)


def latex_table(rows: list[list[str]], header: list[str], *, caption: str, label: str,
                align: str | None = None, group_rule_before: set[int] | None = None,
                notes: str | None = None, size: str = r"\small") -> str:
    """Assemble a booktabs table. ``rows`` cells must already be LaTeX-formatted."""
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [r"\begin{table}[t]", r"\centering", size,
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if group_rule_before and i in group_rule_before and i > 0:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\smallskip\footnotesize {notes}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def bold_best(cells: list[str], values: list[float], *, higher_is_better: bool = True,
              eligible: list[bool] | None = None) -> list[str]:
    """Bold the cell(s) holding the best numeric value among eligible rows."""
    eligible = eligible or [True] * len(values)
    cand = [v for v, e in zip(values, eligible, strict=True)
            if e and v is not None and not math.isnan(v)]
    if not cand:
        return cells
    best = max(cand) if higher_is_better else min(cand)
    return [rf"\textbf{{{c}}}" if e and v is not None and not math.isnan(v)
            and abs(v - best) < 1e-12 else c
            for c, v, e in zip(cells, values, eligible, strict=True)]


def write_table(path: Path, tex: str, df: pd.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tex").write_text(tex)
    if df is not None:
        df.to_csv(path.with_suffix(".csv"), index=False)
