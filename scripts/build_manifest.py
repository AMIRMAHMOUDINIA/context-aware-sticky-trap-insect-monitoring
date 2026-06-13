#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ALIASES = {
    "species": {
        "tribolium_castaneum": [
            "tribolium",
            "castaneum",
            "red flour beetle",
            "red_flour_beetle",
            "red-flour-beetle",
        ],
        "sitophilus_oryzae": [
            "sitophilus",
            "oryzae",
            "rice weevil",
            "rice_weevil",
            "rice-weevil",
        ],
    },
    "device": {
        "dslr": ["dslr", "digital single lens", "digital_single_lens"],
        "webcam": ["webcam", "web cam", "web_cam"],
        "smartphone": ["smartphone", "smart phone", "mobile", "phone"],
    },
    "trap_color": {
        "blue": ["blue"],
        "yellow": ["yellow"],
        "white": ["white"],
        "transparent": ["transparent", "clear"],
        "mixed": ["mixed", "mix colour", "mix_color", "four colour", "four_color"],
    },
    "split": {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation"],
        "test": ["test", "testing"],
    },
}


def _normalise(value: str) -> str:
    value = value.lower().replace("\\", "/")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _infer(text: str, field: str) -> str:
    normalised = f" {_normalise(text)} "
    matches: list[str] = []
    for canonical, aliases in ALIASES[field].items():
        for alias in aliases:
            alias_norm = _normalise(alias)
            if f" {alias_norm} " in normalised:
                matches.append(canonical)
                break
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else ""


def _load_overrides(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    if "image_path" not in frame.columns:
        raise ValueError("Override CSV must contain image_path.")
    return frame.set_index("image_path")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image metadata from a directory tree.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overrides", default=None)
    args = parser.parse_args()

    root = Path(args.image_root)
    if not root.exists():
        raise FileNotFoundError(root)
    overrides = _load_overrides(args.overrides)
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        record = {
            "image_path": relative,
            "species": _infer(relative, "species"),
            "device": _infer(relative, "device"),
            "trap_color": _infer(relative, "trap_color"),
            "split": _infer(relative, "split"),
            "group_id": "",
        }
        if overrides is not None and relative in overrides.index:
            for column in ["species", "device", "trap_color", "split", "group_id"]:
                if column in overrides.columns and pd.notna(overrides.at[relative, column]):
                    record[column] = str(overrides.at[relative, column]).strip()
        missing = [field for field in ["species", "device", "trap_color"] if not record[field]]
        record["parse_status"] = "ok" if not missing else "review"
        record["parse_notes"] = ",".join(missing)
        records.append(record)

    if not records:
        raise RuntimeError(f"No supported images found below {root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows to {output}")
    print(frame["parse_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
