import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim

from datasets.cifar import get_cifar10_loaders
from models.cnns.resnet import resnet34
from training.engine import Trainer
from utils.torch_utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg.dataset.root).mkdir(exist_ok=True)

    train_loader, val_loader = get_cifar10_loaders(
        root=Path(cfg.dataset.root),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        mean=cfg.dataset.mean,
        std=cfg.dataset.std
    )

    model = resnet34(num_classes=cfg.dataset.num_classes, color_channels=cfg.dataset.color_channels)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.scheduler.mode,
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
        threshold=cfg.scheduler.threshold,
        threshold_mode=cfg.scheduler.threshold_mode,
        cooldown=cfg.scheduler.cooldown,
        min_lr=cfg.scheduler.min_lr,
        eps=cfg.scheduler.eps,
    )

    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, cfg, device, scheduler)
    trainer.fit()


if __name__ == "__main__":
    main()
