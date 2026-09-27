"""Step 2 of the workflow: Data Profiling.

Builds a structured profile of a DataFrame that is used twice:
  1. rendered in the UI as a profiling dashboard, and
  2. condensed into a text summary that is given to the LLM as context.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
SUPPORTED_TYPES = ["csv", "tsv", "txt", "xlsx", "json", "parquet"]


def load_data(file_name: str, raw: bytes) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame based on its extension."""
    ext = file_name.rsplit(".", 1)[-1].lower()
    buffer = io.BytesIO(raw)

    if ext in ("csv", "txt"):
        # sep=None lets pandas sniff the delimiter (comma, semicolon, tab, ...)
        try:
            return pd.read_csv(buffer, sep=None, engine="python")
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=None, engine="python", encoding="latin-1")
    if ext == "tsv":
        return pd.read_csv(buffer, sep="\t")
    if ext == "xlsx":
        return pd.read_excel(buffer)
    if ext == "json":
        try:
            return pd.read_json(buffer)
        except ValueError:
            buffer.seek(0)
            return pd.read_json(buffer, lines=True)
    if ext == "parquet":
        return pd.read_parquet(buffer)
    raise ValueError(f"Unsupported file type: .{ext}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from headers and try to parse date-like text columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue
        looks_like_date = sample.str.match(r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}").mean() > 0.8
        if looks_like_date:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[col] = parsed
    return df


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    n_missing: int
    missing_pct: float
    n_duplicates: int
    memory_mb: float
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    boolean_cols: list[str]
    columns: pd.DataFrame          # one row per column
    numeric_stats: pd.DataFrame    # describe() + skew for numeric columns
    correlations: pd.DataFrame     # Pearson correlation matrix
    alerts: list[dict] = field(default_factory=list)

    @property
    def quality_score(self) -> int:
        """A simple 0-100 health score: penalise missing cells, duplicates and alerts."""
        score = 100.0
        score -= min(self.missing_pct, 40)
        score -= min(self.n_duplicates / max(self.n_rows, 1) * 100, 20)
        score -= sum(4 if a["level"] == "warning" else 1 for a in self.alerts)
        return int(max(0, min(100, round(score))))


def _column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def profile_data(df: pd.DataFrame) -> DataProfile:
    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    total_cells = max(n_rows * n_cols, 1)

    kinds = {c: _column_kind(df[c]) for c in df.columns}
    numeric_cols = [c for c, k in kinds.items() if k == "numeric"]
    categorical_cols = [c for c, k in kinds.items() if k == "categorical"]
    datetime_cols = [c for c, k in kinds.items() if k == "datetime"]
    boolean_cols = [c for c, k in kinds.items() if k == "boolean"]

    rows = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        try:
            n_unique = int(s.nunique(dropna=True))
        except TypeError:  # unhashable values such as lists
            n_unique = int(s.astype(str).nunique(dropna=True))
        top = non_null.astype(str).value_counts().index[0] if not non_null.empty else "—"
        rows.append(
            {
                "column": c,
                "type": kinds[c],
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_%": round(s.isna().mean() * 100, 2),
                "unique": n_unique,
                "unique_%": round(n_unique / max(n_rows, 1) * 100, 2),
                "most_frequent": str(top)[:40],
            }
        )
    columns = pd.DataFrame(rows)

    if numeric_cols:
        numeric_stats = df[numeric_cols].describe().T
        numeric_stats["skew"] = df[numeric_cols].skew(numeric_only=True)
        numeric_stats = numeric_stats.round(3)
    else:
        numeric_stats = pd.DataFrame()

    correlations = df[numeric_cols].corr(numeric_only=True).round(3) if len(numeric_cols) > 1 else pd.DataFrame()

    profile = DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        n_missing=n_missing,
        missing_pct=round(n_missing / total_cells * 100, 2),
        n_duplicates=int(df.duplicated().sum()) if n_rows else 0,
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        boolean_cols=boolean_cols,
        columns=columns,
        numeric_stats=numeric_stats,
        correlations=correlations,
    )
    profile.alerts = _build_alerts(df, profile)
    return profile


def _build_alerts(df: pd.DataFrame, p: DataProfile) -> list[dict]:
    alerts: list[dict] = []

    def add(level: str, column: str, message: str) -> None:
        alerts.append({"level": level, "column": column, "message": message})

    if p.n_duplicates:
        add("warning", "—", f"{p.n_duplicates:,} duplicate rows ({p.n_duplicates / p.n_rows:.1%} of data).")

    for _, r in p.columns.iterrows():
        col = r["column"]
        if r["missing_%"] >= 50:
            add("warning", col, f"{r['missing_%']:.1f}% missing values.")
        elif r["missing_%"] >= 5:
            add("info", col, f"{r['missing_%']:.1f}% missing values.")
        if r["unique"] <= 1 and p.n_rows > 1:
            add("warning", col, "Constant column — has a single value.")
        elif r["type"] == "categorical" and r["unique_%"] > 90 and p.n_rows > 20:
            add("info", col, "High cardinality — looks like an ID or free text.")

    for col in p.numeric_cols:
        skew = p.numeric_stats.loc[col, "skew"] if col in p.numeric_stats.index else np.nan
        if pd.notna(skew) and abs(skew) > 2:
            add("info", col, f"Highly skewed distribution (skew = {skew:.2f}).")
        s = df[col].dropna()
        if len(s) > 10:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                if n_out / len(s) > 0.05:
                    add("info", col, f"{n_out:,} outliers by the IQR rule ({n_out / len(s):.1%}).")

    if not p.correlations.empty:
        corr = p.correlations.abs().where(np.triu(np.ones(p.correlations.shape, dtype=bool), k=1))
        for (a, b), v in corr.stack().items():
            if v >= 0.9:
                add("info", f"{a} ↔ {b}", f"Very strong correlation (|r| = {v:.2f}).")
    return alerts


# --------------------------------------------------------------------------- #
# LLM context
# --------------------------------------------------------------------------- #
def profile_to_text(df: pd.DataFrame, p: DataProfile, max_cols: int = 60) -> str:
    """Condense the profile into a compact text block for the LLM system prompt."""
    lines = [
        f"Shape: {p.n_rows:,} rows x {p.n_cols} columns",
        f"Missing cells: {p.n_missing:,} ({p.missing_pct}%) | Duplicate rows: {p.n_duplicates:,}",
        "",
        "Columns (name | type | dtype | missing% | unique | most frequent):",
    ]
    for _, r in p.columns.head(max_cols).iterrows():
        lines.append(
            f"- {r['column']} | {r['type']} | {r['dtype']} | {r['missing_%']}% | {r['unique']} | {r['most_frequent']}"
        )
    if p.n_cols > max_cols:
        lines.append(f"... and {p.n_cols - max_cols} more columns")

    if not p.numeric_stats.empty:
        stats = p.numeric_stats[["mean", "std", "min", "50%", "max"]].head(max_cols)
        lines += ["", "Numeric summary:", stats.to_string()]

    if p.alerts:
        lines += ["", "Data quality alerts:"]
        lines += [f"- [{a['level']}] {a['column']}: {a['message']}" for a in p.alerts[:25]]

    lines += ["", "First 5 rows:", df.head(5).to_string(max_colwidth=30)]
    return "\n".join(lines)
