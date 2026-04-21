from pathlib import Path
import pandas as pd

def load_csv(uploaded_file=None, path: str | None = None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if path is None:
        raise ValueError("Provide either uploaded_file or path")
    return pd.read_csv(path)

def validate_columns(df: pd.DataFrame, selected_attributes: list[str]) -> None:
    missing = [c for c in selected_attributes if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def prepare_ranked_df(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    if rank_col not in df.columns:
        raise ValueError(f"Rank column '{rank_col}' not found")

    return df.sort_values(rank_col, ascending=True).reset_index(drop=True)