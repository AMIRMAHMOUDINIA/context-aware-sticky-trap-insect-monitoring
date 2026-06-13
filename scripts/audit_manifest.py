#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside_root(root: Path, relative_path: object) -> Path:
    root = root.expanduser().resolve()
    relative = Path(str(relative_path).strip())
    if relative.is_absolute():
        raise ValueError(f"Manifest image_path must be relative: {relative}")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Manifest image_path escapes image_root: {relative_path!r}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit labels and image readability.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna("")
    root = Path(args.image_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    required = ["image_path", "species", "device", "trap_color"]
    missing_columns = [column for column in required if column not in manifest.columns]
    if missing_columns:
        raise ValueError(f"Missing manifest columns: {missing_columns}")

    image_records: list[dict] = []
    unsafe_records: list[dict[str, str]] = []
    for row in manifest.itertuples(index=False):
        try:
            path = _resolve_inside_root(root, row.image_path)
        except ValueError as exc:
            unsafe_records.append({"image_path": str(row.image_path), "error": str(exc)})
            image_records.append(
                {
                    "image_path": row.image_path,
                    "exists": False,
                    "readable": False,
                    "width": None,
                    "height": None,
                    "mode": "",
                    "error": str(exc),
                    "sha256": "",
                }
            )
            continue

        record = {
            "image_path": row.image_path,
            "exists": path.is_file(),
            "readable": False,
            "width": None,
            "height": None,
            "mode": "",
            "error": "",
            "sha256": "",
        }
        if path.is_file():
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    record["width"], record["height"] = image.size
                    record["mode"] = image.mode
                    record["readable"] = True
                record["sha256"] = _sha256(path)
            except Exception as exc:  # Audit should record all decoder failures.
                record["error"] = str(exc)
        image_records.append(record)

    image_audit = pd.DataFrame(image_records)
    image_audit.to_csv(output / "image_audit.csv", index=False)
    pd.DataFrame(unsafe_records, columns=["image_path", "error"]).to_csv(
        output / "unsafe_paths.csv", index=False
    )

    unparsed_mask = manifest[["species", "device", "trap_color"]].eq("").any(axis=1)
    manifest.loc[unparsed_mask].to_csv(output / "unparsed_rows.csv", index=False)
    manifest.groupby(["species", "device", "trap_color"], dropna=False).size().rename(
        "n_images"
    ).reset_index().to_csv(output / "label_distribution.csv", index=False)

    duplicate_paths = manifest[manifest["image_path"].duplicated(keep=False)]
    duplicate_paths.to_csv(output / "duplicate_paths.csv", index=False)
    exact_duplicates = image_audit[
        image_audit["sha256"].ne("") & image_audit["sha256"].duplicated(keep=False)
    ].sort_values(["sha256", "image_path"])
    exact_duplicates.to_csv(output / "exact_duplicate_files.csv", index=False)
    lines = [
        "# Data audit",
        "",
        f"- Manifest rows: **{len(manifest)}**",
        f"- Missing or unreadable files: **{int((~image_audit['readable']).sum())}**",
        f"- Unsafe or escaping paths: **{len(unsafe_records)}**",
        f"- Rows requiring label review: **{int(unparsed_mask.sum())}**",
        f"- Duplicate image paths: **{len(duplicate_paths)}**",
        f"- Files belonging to exact-duplicate groups: **{len(exact_duplicates)}**",
        "",
        "Review `unparsed_rows.csv`, `image_audit.csv`, `unsafe_paths.csv`, and "
        "`exact_duplicate_files.csv` before training.",
    ]
    (output / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
