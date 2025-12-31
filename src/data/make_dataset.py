import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd

from .clean_transform import SplitConfig, build_reference_stats, clean_transform, save_dataframe, save_json


def _download_from_s3(s3_uri: str, dest: Path) -> Path:
    p = urlparse(s3_uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(dest))
    return dest


def _read_csv_anywhere(input_path: str) -> pd.DataFrame:
    if input_path.startswith("s3://"):
        tmp = Path("data/raw/_tmp_download.csv")
        _download_from_s3(input_path, tmp)
        return pd.read_csv(tmp)
    return pd.read_csv(input_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Chemin CSV local ou S3 (s3://bucket/key)")
    parser.add_argument("--out-dir", default="data/processed", help="Dossier de sortie")
    parser.add_argument("--fmt", default="parquet", choices=["parquet", "csv"], help="Format de sortie")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sort-col", default="id", help="Colonne de tri pour split (ex: id). Vide = ordre fichier.")
    parser.add_argument("--stats-path", default="models/reference_stats.json")
    args = parser.parse_args()

    df = _read_csv_anywhere(args.input)
    df = clean_transform(df)

    sort_col = args.sort_col.strip() if args.sort_col is not None else ""
    sort_col = sort_col if sort_col else None

    train_df, test_df = split_train_test(df, SplitConfig(test_size=args.test_size, sort_col=sort_col))

    out_dir = Path(args.out_dir)
    save_dataframe(train_df, out_dir / f"train.{args.fmt}", fmt=args.fmt)
    save_dataframe(test_df, out_dir / f"test.{args.fmt}", fmt=args.fmt)

    stats = build_reference_stats(train_df)
    save_json(stats, Path(args.stats_path))

    print(f"OK | train={len(train_df)} | test={len(test_df)} | fraud_rate_train={stats['class_balance']['fraud_rate']:.6f}")
    print(f"Saved: {out_dir / f'train.{args.fmt}'}")
    print(f"Saved: {out_dir / f'test.{args.fmt}'}")
    print(f"Saved: {args.stats_path}")


def split_train_test(df, cfg):
    return __import__("src.data.clean_transform", fromlist=["split_train_test"]).split_train_test(df, cfg)


if __name__ == "__main__":
    main()
