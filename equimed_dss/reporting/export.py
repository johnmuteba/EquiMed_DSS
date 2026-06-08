"""Render a DataFrame to markdown / LaTeX / HTML with consistent rounding."""
import os
from typing import Optional

import pandas as pd

_FORMATS = ("markdown", "latex", "html")


def export_table(
    df: pd.DataFrame,
    fmt: str = "markdown",
    path: Optional[str] = None,
    decimals: int = 3,
) -> Optional[str]:
    """Render a table for slides or the manuscript.

    Mirrors the pandas ``to_csv`` convention: when ``path`` is given the table is
    written to that file and the function returns ``None`` (so a notebook cell
    does not echo a large raw string); when ``path`` is ``None`` the rendered
    string is returned. To view a table inline, call without ``path`` and wrap in
    ``print(...)``, for example ``print(export_table(df, fmt="markdown"))``.

    Args:
        df: any tidy DataFrame (e.g. from equimed_dss.reporting.tables).
        fmt: one of "markdown", "latex", "html".
        path: if given, write the rendered string to this path (parent
            directories are created automatically) and return None.
        decimals: rounding applied to numeric columns before rendering.

    Returns:
        The rendered string when ``path`` is None, otherwise None.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"Unknown fmt {fmt!r}; use one of {_FORMATS}.")

    rounded = df.copy()
    num_cols = rounded.select_dtypes(include="number").columns
    rounded[num_cols] = rounded[num_cols].round(decimals)

    if fmt == "markdown":
        rendered = rounded.to_markdown(index=False)
    elif fmt == "latex":
        rendered = rounded.to_latex(index=False)
    else:  # html
        rendered = rounded.to_html(index=False)

    if path is not None:
        # Create the parent directory if it does not exist, so callers can
        # write to e.g. "results/geographic.md" without pre-creating "results/".
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(rendered)
        return None
    return rendered
