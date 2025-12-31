import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    sort_col: Optional[str] = "id"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Class" not in df.columns:
        raise ValueError("Colonne cible 'Class' introuvable.")
    if "Amount" not in df.columns:
        raise ValueError("Colonne 'Amount' introuvable.")
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
    return df


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Amount"])
    df = df[df["Amount"] >= 0]
    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"].astype(float))
    return df


def split_train_test(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if cfg.sort_col and cfg.sort_col in df.columns:
        df = df.sort_values(cfg.sort_col, kind="mergesort")
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    n_test = int(np.ceil(cfg.test_size * n))
    n_train = n - n_test
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)
    return train_df, test_df


def build_reference_stats(df: pd.DataFrame, cols=("Amount", "log_amount")) -> Dict:
    out = {"n_rows": int(len(df)), "columns": {}}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
        out["columns"][c] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "quantiles": {str(k): float(v) for k, v in q.items()},
        }
    out["class_balance"] = {
        "fraud_rate": float((df["Class"] == 1).mean()),
        "n_fraud": int((df["Class"] == 1).sum()),
        "n_non_fraud": int((df["Class"] == 0).sum()),
    }
    return out


def save_dataframe(df: pd.DataFrame, path: Path, fmt: str = "parquet") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("fmt doit être 'parquet' ou 'csv'.")


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_columns(df)
    df = _cast_types(df)
    df = _basic_clean(df)
    df = _feature_engineering(df)
    return df
