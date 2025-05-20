from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from omegaconf import DictConfig

from utils.torch_utils import build_transforms


def get_cifar10_loaders(root: Path, batch_size: int, num_workers: int, transform_cfg: DictConfig):
    train_transforms = build_transforms(transform_cfg.train)
    val_transforms = build_transforms(transform_cfg.val)

    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transforms)
    val_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
