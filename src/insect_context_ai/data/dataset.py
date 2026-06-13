from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .encoders import ContextEncoder, LabelEncoder


class InsectImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        target_column: str,
        context_columns: list[str],
        label_encoder: LabelEncoder,
        context_encoder: ContextEncoder,
        transform: Any,
    ) -> None:
        self.frame = frame.reset_index(drop=False).rename(columns={"index": "_source_index"})
        self.target_column = target_column
        self.context_columns = context_columns
        self.label_encoder = label_encoder
        self.context_encoder = context_encoder
        self.transform = transform
        self.targets = self.label_encoder.transform(self.frame[target_column])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = Path(row["_absolute_path"])
        try:
            with Image.open(path) as image:
                image_tensor = self.transform(image.copy())
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {path}: {exc}") from exc

        context = self.context_encoder.transform_row(row, self.context_columns)
        return {
            "image": image_tensor,
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "context": torch.tensor(context, dtype=torch.long),
            "source_index": int(row["_source_index"]),
            "image_path": str(row["image_path"]),
        }
