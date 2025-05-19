from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar_10_loaders(root: Path, batch_size: int, num_workers: int, mean: list, std: list):
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_t)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
