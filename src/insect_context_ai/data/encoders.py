from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass
class LabelEncoder:
    class_to_idx: dict[str, int]

    @classmethod
    def fit(cls, values: Iterable[object]) -> LabelEncoder:
        labels = sorted({str(value).strip() for value in values if str(value).strip()})
        if len(labels) < 2:
            raise ValueError(f"At least two target classes are required; found {labels}.")
        return cls({label: index for index, label in enumerate(labels)})

    def transform(self, values: Iterable[object]) -> list[int]:
        encoded: list[int] = []
        for value in values:
            key = str(value).strip()
            if key not in self.class_to_idx:
                raise ValueError(f"Unknown target label: {key!r}")
            encoded.append(self.class_to_idx[key])
        return encoded

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {index: label for label, index in self.class_to_idx.items()}


@dataclass
class ContextEncoder:
    """Encode categorical context with 0 reserved for missing/unseen values."""

    mappings: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> ContextEncoder:
        mappings: dict[str, dict[str, int]] = {}
        for column in columns:
            if column not in frame.columns:
                raise ValueError(f"Context column not found: {column}")
            values = sorted(
                {
                    str(value).strip().lower()
                    for value in frame[column].fillna("")
                    if str(value).strip()
                }
            )
            mappings[column] = {value: index + 1 for index, value in enumerate(values)}
        return cls(mappings)

    def transform_row(self, row: pd.Series, columns: list[str]) -> list[int]:
        result: list[int] = []
        for column in columns:
            raw = str(row.get(column, "")).strip().lower()
            result.append(self.mappings.get(column, {}).get(raw, 0))
        return result

    def cardinalities(self, columns: list[str]) -> list[int]:
        return [len(self.mappings.get(column, {})) + 1 for column in columns]
