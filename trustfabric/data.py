from __future__ import annotations
from pathlib import Path
import zipfile
import unicodedata
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import pandas as pd

LABEL_MAP = {
    "pants-fire": "false", "false": "false",
    "barely-true": "mixed", "half-true": "mixed",
    "mostly-true": "true", "true": "true",
    # Original LIAR numeric labels: 0=pants-fire, 1=false, 2=barely-true,
    # 3=half-true, 4=mostly-true, 5=true.
    "0": "false", "1": "false", "2": "mixed",
    "3": "mixed", "4": "true", "5": "true",
}
OFFICIAL_LIAR_URL = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"


def load_csv(path: str | Path, text_column: str = "statement", label_column: str = "label") -> pd.DataFrame:
    """Load a LIAR-style CSV, normalize text/labels, and preserve source splits."""
    df = pd.read_csv(path)
    if text_column not in df or label_column not in df:
        raise ValueError(f"CSV must contain columns {text_column!r} and {label_column!r}; found {list(df.columns)}")
    original_rows = len(df)
    columns = [text_column, label_column]
    if "split" in df.columns and "split" not in columns:
        columns.append("split")
    df = df[columns].rename(columns={text_column: "text", label_column: "label"})
    df["text"] = df["text"].fillna("").astype(str).map(
        lambda value: unicodedata.normalize("NFKC", value)
    )
    df["text"] = df["text"].str.replace(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip().map(lambda x: LABEL_MAP.get(x, x))
    df = df[(df.text.str.len() > 0) & df.label.isin(["false", "mixed", "true"])]
    if "split" in df.columns:
        split_map = {"train": "train", "valid": "valid", "validation": "valid", "dev": "valid", "test": "test"}
        original_splits = df["split"].astype(str).str.lower().str.strip()
        normalized_splits = original_splits.map(split_map)
        if normalized_splits.isna().any():
            bad_splits = sorted(original_splits[normalized_splits.isna()].unique().tolist())
            raise ValueError(f"Unsupported dataset split values: {bad_splits}")
        df["split"] = normalized_splits
    before_deduplication = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable rows found; provide at least 30 labeled statements for a meaningful split.")
    df = df.reset_index(drop=True)
    df.attrs["original_rows"] = int(original_rows)
    df.attrs["removed_rows"] = int(original_rows - len(df))
    df.attrs["duplicate_text_rows"] = int(before_deduplication - len(df))
    return df


def convert_official_liar_archive(archive_path: str | Path, output_path: str | Path) -> Path:
    """Convert the official archive while retaining its original data partitions."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        frames = []
        for name in ("train.tsv", "valid.tsv", "test.tsv"):
            if name not in zf.namelist():
                continue
            frame = pd.read_csv(zf.open(name), sep="\t", header=None)
            if frame.shape[1] < 3:
                continue
            frame = frame[[1, 2]].rename(columns={1: "label", 2: "statement"})
            frame["split"] = {"train.tsv": "train", "valid.tsv": "valid", "test.tsv": "test"}[name]
            frames.append(frame)
        if not frames:
            raise RuntimeError("The official LIAR archive contains no train/valid/test TSV files.")
        combined = pd.concat(frames, ignore_index=True)
        combined["statement"] = combined["statement"].fillna("").astype(str).str.strip()
        combined["label"] = combined["label"].astype(str).str.lower().str.strip().map(lambda value: LABEL_MAP.get(value, value))
        combined = combined[(combined["statement"].str.len() > 0) & combined["label"].isin(["false", "mixed", "true"])].reset_index(drop=True)
        combined.to_csv(dest, index=False)
    return dest


def download_official_liar_csv(output_path: str | Path) -> Path:
    """Download and convert the official LIAR archive, preserving source splits."""
    dest = Path(output_path)
    cached_archive = dest.with_name("liar_dataset.zip")
    if cached_archive.exists():
        return convert_official_liar_archive(cached_archive, dest)
    with TemporaryDirectory(prefix="liar_") as tmpdir:
        archive = Path(tmpdir) / "liar_dataset.zip"
        urlretrieve(OFFICIAL_LIAR_URL, archive)
        return convert_official_liar_archive(archive, dest)


def download_hf_dataset(output_path: str | Path) -> Path:
    """Download the public LIAR dataset through Hugging Face Datasets.

    Dataset repository availability can change. If that fails, fall back to the
    official LIAR archive from the dataset's original host and convert the TSVs.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install optional dataset downloader with: pip install datasets") from e
    errors = []
    for repo in ("ucsbnlp/liar", "liar"):
        try:
            ds = load_dataset(repo)
            parts = []
            for split in ds.keys():
                frame = ds[split].to_pandas()
                text_col = next((c for c in ("statement", "text", "claim") if c in frame.columns), None)
                label_col = next((c for c in ("label", "labels", "veracity") if c in frame.columns), None)
                if text_col and label_col:
                    part = frame[[text_col, label_col]].rename(columns={text_col: "statement", label_col: "label"})
                    part["split"] = "valid" if split in {"validation", "dev"} else split
                    parts.append(part)
            if parts:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(parts, ignore_index=True).to_csv(out, index=False)
                return out
            errors.append(f"{repo}: expected text/label columns not found")
        except Exception as e:
            errors.append(f"{repo}: {type(e).__name__}: {e}")
    try:
        return download_official_liar_csv(output_path)
    except Exception as archive_error:
        raise RuntimeError(
            "Could not automatically fetch LIAR from known Hugging Face IDs or the official archive. "
            + " | ".join(errors)
            + f" | official archive: {archive_error}. Download a LIAR-format CSV and pass --csv path."
        ) from archive_error
