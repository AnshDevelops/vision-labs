from typing import Optional


class EarlyStopper:
    def __init__(self, mode="min", patience: int = 10, min_delta: float = 1e-4):
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        improvement = (current_score < self.best_score - self.min_delta) if self.mode == "min" \
            else (current_score > self.best_score + self.min_delta)
        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
