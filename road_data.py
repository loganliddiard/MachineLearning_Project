from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


REQUIRED_CSV_COLUMNS = {"image_id", "split", "label"}


@dataclass(frozen=True)
class DatasetPaths:
    data_root: str = "UDOT WINTER ROAD CONDITIONS.v1i.folder"

    @property
    def csv_path(self) -> str:
        return os.path.join(self.data_root, "dataset_index.csv")


def load_index_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"dataset_index.csv missing columns: {sorted(missing)}")
    return df


def build_label_mapping(train_df: pd.DataFrame) -> Dict[str, int]:
    labels = sorted(train_df["label"].astype(str).unique().tolist())
    if not labels:
        raise ValueError("No labels found in TRAIN split.")
    return {label: idx for idx, label in enumerate(labels)}


class RoadConditionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        split: str,
        label_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        verify_files: bool = True,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.label_to_idx = label_to_idx
        self.transform = transform

        split_df = df[df["split"].astype(str) == split].copy()
        if len(split_df) == 0:
            raise ValueError(f"No rows found for split='{split}' in dataset_index.csv")

        split_df["image_id"] = split_df["image_id"].astype(str)
        split_df["label"] = split_df["label"].astype(str)
        self.df = split_df.reset_index(drop=True)

        if verify_files:
            self._verify_some_files(max_to_check=25)

    def _resolve_path(self, image_id: str, label: str) -> str:
        # Matches folder layout assumed by existing scripts:
        # {data_root}/{split}/{label}/{image_id}
        return os.path.join(self.data_root, self.split, label, image_id)

    def _verify_some_files(self, max_to_check: int = 25) -> None:
        checked = 0
        missing: List[str] = []
        for _, row in self.df.iterrows():
            path = self._resolve_path(row["image_id"], row["label"])
            if not os.path.exists(path):
                missing.append(path)
            checked += 1
            if checked >= max_to_check:
                break

        if missing:
            missing_preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                "Some image files could not be found. "
                "Expected layout: {data_root}/{split}/{label}/{image_id}.\n"
                f"Checked {checked} samples in split '{self.split}'. Missing examples:\n{missing_preview}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]
        image_path = self._resolve_path(image_id, label)

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

        y = self.label_to_idx[label]
        return img, y


def default_transforms(
    ensure_224: bool = True,
) -> transforms.Compose:
    # Keep this intentionally minimal to match current workflow:
    # preprocessing resizes to 224x224 already; this is a safety net.
    t: List[object] = []
    if ensure_224:
        t.append(transforms.Resize((224, 224)))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def make_dataloaders(
    *,
    data_root: str,
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    ensure_224: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    df = load_index_csv(csv_path)

    train_df = df[df["split"].astype(str) == "train"].copy()
    label_to_idx = build_label_mapping(train_df)

    transform = default_transforms(ensure_224=ensure_224)

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = RoadConditionDataset(df, data_root=data_root, split="train", label_to_idx=label_to_idx, transform=transform)
    valid_ds = RoadConditionDataset(df, data_root=data_root, split="valid", label_to_idx=label_to_idx, transform=transform)
    test_ds = RoadConditionDataset(df, data_root=data_root, split="test", label_to_idx=label_to_idx, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    return train_dl, valid_dl, test_dl, label_to_idx
