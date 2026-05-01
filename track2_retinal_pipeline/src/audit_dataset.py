#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset audit for the MMRDR CFP, UWF, and 2D OCT folders."""

from __future__ import annotations

import argparse
import ast
import hashlib
import logging
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


LOGGER = logging.getLogger("retinal_audit")

MODALITIES = {
    "CFP": {"folder": "MMRDR-CFP", "csv": "FP.csv", "classes": 5},
    "UWF": {"folder": "MMRDR-UWF", "csv": "UWF.csv", "classes": 5},
    "OCT": {"folder": "MMRDR-OCT", "csv": "OCT.csv", "classes": 3},
}

LESION_NAMES = [
    "microaneurysm",
    "hard_exudate",
    "intraretinal_hemorrhage",
    "vb_irma",
    "neovascularization",
    "vitreous_hemorrhage",
    "retinal_detachment",
]


def split_from_image_path(path: str) -> str:
    name = Path(path).name.lower()
    if name.startswith("tr"):
        return "train"
    if name.startswith("ts"):
        return "test"
    return "unknown"


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lesion(value: str) -> list[int] | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list) or len(parsed) != 7:
        return None
    return [int(v) for v in parsed]


def load_modality(data_root: Path, modality: str) -> pd.DataFrame:
    spec = MODALITIES[modality]
    base = data_root / spec["folder"]
    df = pd.read_csv(base / spec["csv"])
    df["modality"] = modality
    df["image_path"] = df["image"].map(lambda p: str(base / p))
    df["image_exists"] = df["image_path"].map(lambda p: Path(p).exists())
    df["split"] = df["image"].map(split_from_image_path)
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce").astype("Int64")
    return df


def create_sample_grid(df: pd.DataFrame, out_path: Path, title: str, max_per_class: int = 4) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for grade in sorted(df["grade"].dropna().unique().tolist()):
        subset = df[(df["grade"] == grade) & (df["image_exists"])].head(max_per_class)
        for _, row in subset.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                img = ImageOps.contain(img, (180, 180))
                canvas = Image.new("RGB", (200, 220), "white")
                canvas.paste(img, ((200 - img.width) // 2, 8))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 192), f"grade {grade}", fill="black")
                thumbs.append((str(grade), canvas))
            except Exception as exc:
                LOGGER.warning("Could not render %s: %s", row["image_path"], exc)
    if not thumbs:
        return
    cols = max_per_class
    rows = int((len(thumbs) + cols - 1) / cols)
    grid = Image.new("RGB", (cols * 200, rows * 220 + 36), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((8, 8), title, fill="black")
    for i, (_, img) in enumerate(thumbs):
        x = (i % cols) * 200
        y = 36 + (i // cols) * 220
        grid.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def audit_modality(data_root: Path, modality: str, out_dir: Path, hash_duplicates: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_modality(data_root, modality)
    spec = MODALITIES[modality]
    base = data_root / spec["folder"]
    jpg_files = sorted((base / "img").glob("*.jpg"))
    jpg_names = {f"img/{p.name}" for p in jpg_files}
    csv_names = set(df["image"].astype(str))

    missing_files = df[~df["image_exists"]].copy()
    extra_images = sorted(jpg_names - csv_names)
    bad_split = df[df["split"] == "unknown"].copy()
    bad_grade = df[~df["grade"].isin(range(spec["classes"]))].copy()

    dist = (
        df.groupby(["modality", "split", "grade"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["modality", "split", "grade"])
    )

    duplicate_rows = []
    if hash_duplicates:
        hash_to_paths: dict[str, list[str]] = {}
        for image_path in df.loc[df["image_exists"], "image_path"]:
            path = Path(image_path)
            h = file_md5(path)
            hash_to_paths.setdefault(h, []).append(str(path))
        for h, paths in hash_to_paths.items():
            if len(paths) > 1:
                duplicate_rows.append({"modality": modality, "hash": h, "count": len(paths), "paths": " | ".join(paths)})
    duplicates = pd.DataFrame(duplicate_rows)

    if "lesion" in df.columns and modality in {"CFP", "UWF"}:
        lesion_values = df["lesion"].map(parse_lesion)
        lesion_df = pd.DataFrame([v if v is not None else [None] * 7 for v in lesion_values], columns=LESION_NAMES)
        lesion_summary = lesion_df.sum(skipna=True).reset_index()
        lesion_summary.columns = ["lesion", "positive_count"]
        lesion_summary["modality"] = modality
    else:
        lesion_summary = pd.DataFrame(columns=["modality", "lesion", "positive_count"])

    checks = pd.DataFrame(
        [
            {"modality": modality, "check": "csv_rows", "value": len(df), "status": "info"},
            {"modality": modality, "check": "jpg_files", "value": len(jpg_files), "status": "info"},
            {"modality": modality, "check": "missing_image_paths", "value": len(missing_files), "status": "pass" if missing_files.empty else "fail"},
            {"modality": modality, "check": "extra_jpg_not_in_csv", "value": len(extra_images), "status": "pass" if not extra_images else "warn"},
            {"modality": modality, "check": "unknown_split_prefix", "value": len(bad_split), "status": "pass" if bad_split.empty else "fail"},
            {"modality": modality, "check": "invalid_grade", "value": len(bad_grade), "status": "pass" if bad_grade.empty else "fail"},
            {
                "modality": modality,
                "check": "exact_duplicate_images",
                "value": len(duplicates),
                "status": ("pass" if duplicates.empty else "warn") if hash_duplicates else "not_run",
            },
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{modality.lower()}_manifest.csv", index=False)
    dist.to_csv(out_dir / f"{modality.lower()}_class_distribution.csv", index=False)
    missing_files.to_csv(out_dir / f"{modality.lower()}_missing_images.csv", index=False)
    pd.DataFrame({"image": extra_images}).to_csv(out_dir / f"{modality.lower()}_extra_images.csv", index=False)
    duplicates.to_csv(out_dir / f"{modality.lower()}_duplicate_hashes.csv", index=False)
    lesion_summary.to_csv(out_dir / f"{modality.lower()}_lesion_summary.csv", index=False)
    create_sample_grid(df, out_dir / f"{modality.lower()}_sample_grid.jpg", f"{modality} samples by class")
    return checks, dist, lesion_summary


def write_report(out_dir: Path, checks: pd.DataFrame, dist: pd.DataFrame, lesions: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        headers = list(df.columns)
        rows = df.astype(str).values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# Track 2 MMRDR Dataset Audit",
        "",
        "## Integrity Checks",
        "",
        markdown_table(checks),
        "",
        "## Class Distribution",
        "",
        markdown_table(dist),
    ]
    if not lesions.empty:
        lines.extend(["", "## Lesion Label Counts", "", markdown_table(lesions)])
    lines.extend(
        [
            "",
            "## Split Policy",
            "",
            "- `tr*.jpg` files are training/validation candidates.",
            "- `ts*.jpg` files are locked test cases.",
            "- Validation splits should be stratified and created only from `tr*.jpg` rows.",
            "- Exact duplicate hashing is optional because it reads every image byte; run with `--hash-duplicates` for the full duplicate audit.",
        ]
    )
    (out_dir / "dataset_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(data_root: Path, out_dir: Path, hash_duplicates: bool = False) -> None:
    all_checks = []
    all_dist = []
    all_lesions = []
    for modality in MODALITIES:
        LOGGER.info("Auditing %s", modality)
        checks, dist, lesions = audit_modality(data_root, modality, out_dir, hash_duplicates=hash_duplicates)
        all_checks.append(checks)
        all_dist.append(dist)
        all_lesions.append(lesions)
    checks_df = pd.concat(all_checks, ignore_index=True)
    dist_df = pd.concat(all_dist, ignore_index=True)
    lesions_df = pd.concat(all_lesions, ignore_index=True)
    checks_df.to_csv(out_dir / "integrity_checks.csv", index=False)
    dist_df.to_csv(out_dir / "class_distribution.csv", index=False)
    lesions_df.to_csv(out_dir / "lesion_summary.csv", index=False)
    write_report(out_dir, checks_df, dist_df, lesions_df)
    LOGGER.info("Dataset audit saved to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the MMRDR retinal image dataset.")
    parser.add_argument("--data-root", default="../Datasets/Paper 2", help="Path containing MMRDR-CFP, MMRDR-UWF, and MMRDR-OCT.")
    parser.add_argument("--out-dir", default="outputs/dataset_audit", help="Output directory.")
    parser.add_argument("--hash-duplicates", action="store_true", help="Compute exact MD5 duplicate-image hashes. Slower but stricter.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    run_audit(Path(args.data_root), Path(args.out_dir), hash_duplicates=args.hash_duplicates)


if __name__ == "__main__":
    main()
