import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: nn.Module, optimizer, criterion, train_loader, val_loader, cfg,
                 device):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.val_loader = train_loader, val_loader
        self.epochs = cfg.training.epochs
        self.log_dir = Path(cfg.logging.log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if cfg.logging.tensorboard else None

        if cfg.logging.use_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0

        pbar = self.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            preds = self.model(imgs)
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}")
        if self.writer:
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch: int):
        self.model.eval()
        correct, total, val_loss = 0, 0, 0

        pbar = self.tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)
        with torch.no_grad():
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)
                val_loss += self.criterion(outputs, labels).item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = val_loss / len(self.val_loader)
            acc = correct / total
            logger.info(f"Epoch {epoch + 1} Val Loss: {avg_loss:.4f} Val Acc: {acc:.4f}")
            if self.writer:
                self.writer.add_scalar("Loss/train", avg_loss, epoch)
                self.writer.add_scalar("Acc/val", acc, epoch)
            return avg_loss

    def fit(self):
        best_acc = 0
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            _, acc = self.validate(epoch)

            if acc > best_acc:
                best_acc = acc
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                }
                save_path = self.log_dir / f"best_epoch{epoch + 1}.pth"
                torch.save(checkpoint, save_path)
                logger.info(f"Best model saved at {save_path} with accuracy: {best_acc:.4f}")
