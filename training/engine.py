import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from training.early_stopping import EarlyStopper

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: nn.Module, optimizer, criterion, train_loader, val_loader, cfg,
                 device, scheduler=None):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.val_loader = train_loader, val_loader
        self.epochs = cfg.training.epochs
        self.log_dir = Path(cfg.logging.log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if cfg.logging.tensorboard else None
        self.scheduler = scheduler

        if cfg.logging.use_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct, total = 0, 0

        pbar = self.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            preds = self.model(imgs)
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            _, predicted = preds.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_train_loss = total_loss / len(self.train_loader)
        train_acc = correct / total
        logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if self.writer:
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
        return avg_train_loss, train_acc

    def validate(self, epoch: int):
        self.model.eval()
        correct, total, total_loss = 0, 0, 0

        pbar = self.tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)
        with torch.no_grad():
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)
                total_loss += self.criterion(outputs, labels).item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            avg_val_loss = total_loss / len(self.val_loader)
            val_acc = correct / total
            logger.info(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.4f}")
            if self.writer:
                self.writer.add_scalar("Loss/val", avg_val_loss, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)
            return avg_val_loss, val_acc

    def fit(self, patience: int, min_delta: float):
        best_acc = 0.0

        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if early_stopper(val_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                }
                save_path = self.log_dir / f"best_epoch{epoch + 1}.pth"
                torch.save(checkpoint, save_path)
                logger.info(f"Best model saved at {save_path} with accuracy: {best_acc:.4f}")
