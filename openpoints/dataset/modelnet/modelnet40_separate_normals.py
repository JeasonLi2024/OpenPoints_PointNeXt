from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..build import DATASETS


def _class_name(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


@DATASETS.register_module()
class ModelNet40SeparateNormals(Dataset):
    """ModelNet40 adapter for separate train/test roots with xyz and normals."""

    def __init__(
        self,
        train_data_dir,
        test_data_dir,
        num_points=1024,
        num_classes=40,
        split="train",
        transform=None,
        val_ratio=0.15,
        split_seed=42,
        normalize_xyz=True,
        use_normals=True,
    ):
        self.train_root = Path(train_data_dir).expanduser()
        self.test_root = Path(test_data_dir).expanduser()
        self.num_points = int(num_points)
        self.num_category = int(num_classes)
        self.split = split.lower()
        self.transform = transform
        self.normalize_xyz = bool(normalize_xyz)
        self.use_normals = bool(use_normals)

        if self.num_points <= 0:
            raise ValueError(f"num_points must be positive, got {self.num_points}")
        if not 0.0 < float(val_ratio) < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        if not self.train_root.is_dir():
            raise FileNotFoundError(f"training data directory not found: {self.train_root}")
        if not self.test_root.is_dir():
            raise FileNotFoundError(f"test data directory not found: {self.test_root}")

        shape_names = self.train_root / "modelnet40_shape_names.txt"
        if not shape_names.is_file():
            raise FileNotFoundError(f"class list not found: {shape_names}")
        self.classes = [
            line.strip() for line in shape_names.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self.classes) != self.num_category:
            raise ValueError(
                f"expected {self.num_category} classes, found {len(self.classes)}"
            )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        if self.split in {"train", "val"}:
            items = self._training_items()
            train_items, val_items = self._stratified_split(
                items, float(val_ratio), int(split_seed)
            )
            self.items = train_items if self.split == "train" else val_items
        elif self.split == "test":
            self.items = self._test_items()
        else:
            raise ValueError(f"unsupported split: {split}")

        if not self.items:
            raise FileNotFoundError(f"no samples found for split={self.split}")

    def _training_items(self):
        manifest = self.train_root / "modelnet40_train.txt"
        if not manifest.is_file():
            raise FileNotFoundError(f"training manifest not found: {manifest}")
        sample_ids = [
            line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        items = []
        for sample_id in sample_ids:
            class_name = _class_name(sample_id)
            if class_name not in self.class_to_idx:
                raise ValueError(f"unknown class '{class_name}' in {manifest}")
            path = self.train_root / class_name / f"{sample_id}.txt"
            if not path.is_file():
                raise FileNotFoundError(f"training sample not found: {path}")
            items.append((path, self.class_to_idx[class_name], sample_id))
        return items

    def _test_items(self):
        items = []
        for class_name in self.classes:
            class_dir = self.test_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"test class directory not found: {class_dir}")
            for path in sorted(class_dir.glob("*.txt")):
                items.append((path, self.class_to_idx[class_name], path.stem))
        return items

    def _stratified_split(self, items, val_ratio, seed):
        train_items, val_items = [], []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_items = [item for item in items if item[1] == class_idx]
            rng = np.random.default_rng(_stable_seed(class_name, seed))
            order = rng.permutation(len(class_items))
            val_count = max(1, int(round(len(class_items) * val_ratio)))
            val_ids = set(order[:val_count].tolist())
            for index, item in enumerate(class_items):
                (val_items if index in val_ids else train_items).append(item)
        return train_items, val_items

    def __len__(self):
        return len(self.items)

    @property
    def num_classes(self):
        return self.num_category

    def __getitem__(self, index):
        path, label, _ = self.items[index]
        points = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 6:
            raise ValueError(
                f"{path} must have shape [N, 6] for x,y,z,nx,ny,nz; "
                f"got {points.shape}"
            )
        if len(points) < self.num_points:
            raise ValueError(
                f"{path} has {len(points)} points, fewer than num_points={self.num_points}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"{path} contains NaN or infinity")

        xyz = points[:, :3].copy()
        normals = points[:, 3:6].copy()

        if self.normalize_xyz:
            xyz -= xyz.mean(axis=0, keepdims=True)
            radius = np.linalg.norm(xyz, axis=1).max()
            if radius > 0:
                xyz /= radius

        normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(normal_norm, 1e-12)

        if self.split == "train":
            # OpenPoints performs the final FPS/random 1024-point selection on GPU.
            order = np.random.permutation(len(xyz))
            xyz, normals = xyz[order], normals[order]

        data = {"pos": xyz, "normal": normals, "y": np.int64(label)}
        if self.transform is not None:
            data = self.transform(data)

        if not torch.is_tensor(data["pos"]):
            data["pos"] = torch.from_numpy(data["pos"]).float()
        if not torch.is_tensor(data["normal"]):
            data["normal"] = torch.from_numpy(data["normal"]).float()
        data["x"] = (
            torch.cat((data["pos"], data["normal"]), dim=1)
            if self.use_normals
            else data["pos"]
        )
        return data
