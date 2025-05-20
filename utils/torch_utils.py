import random, numpy as np, torch

from omegaconf import DictConfig
from torchvision import transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(cfg: DictConfig) -> transforms.Compose:
    ops = []
    for op, kwargs in cfg.items():
        op_cls = getattr(transforms, op)
        ops.append(op_cls(**kwargs))
    return transforms.Compose(ops)
