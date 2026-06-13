from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"image_path", "species"}


def resolve_image_path(root: str | Path, relative_path: object) -> Path:
    """Resolve a manifest path and ensure it remains inside the declared image root."""
    root_path = Path(root).expanduser().resolve()
    relative = Path(str(relative_path).strip())
    if relative.is_absolute():
        raise ValueError(f"Manifest image_path must be relative, not absolute: {relative}")

    candidate = (root_path / relative).resolve()
    if candidate != root_path and root_path not in candidate.parents:
        raise ValueError(
            "Manifest image_path escapes the configured image_root: "
            f"{relative_path!r}"
        )
    return candidate


def load_manifest(path: str | Path, image_root: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    root = Path(image_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")

    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest contains no rows.")

    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["image_path"] = frame["image_path"].astype(str).str.strip()

    empty_paths = frame["image_path"].eq("")
    if empty_paths.any():
        examples = frame.index[empty_paths].tolist()[:5]
        raise ValueError(
            f"Manifest contains {int(empty_paths.sum())} empty image_path values. "
            f"Example row indices: {examples}"
        )

    empty_targets = frame["species"].astype(str).str.strip().eq("")
    if empty_targets.any():
        examples = frame.index[empty_targets].tolist()[:5]
        raise ValueError(
            f"Manifest contains {int(empty_targets.sum())} rows with an empty species label. "
            f"Example row indices: {examples}"
        )

    duplicated = frame["image_path"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "image_path"].head(5).tolist()
        raise ValueError(f"Duplicate image paths found in manifest, for example: {examples}")

    absolute_paths: list[str] = []
    for value in frame["image_path"]:
        absolute_paths.append(str(resolve_image_path(root, value)))
    frame["_absolute_path"] = absolute_paths
    frame["_row_id"] = range(len(frame))

    missing_files = [path for path in frame["_absolute_path"] if not Path(path).is_file()]
    if missing_files:
        examples = missing_files[:5]
        raise FileNotFoundError(
            f"{len(missing_files)} image files referenced by the manifest were not found. "
            f"Examples: {examples}"
        )
    return frame
