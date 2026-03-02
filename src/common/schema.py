"""Lightweight schema validation utilities."""

from __future__ import annotations


def require_columns(df, cols: list[str], table_name: str = "table") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name}: missing required columns: {missing}")
