import torch
from .containers import Module
from typing import List
import torch.nn.functional as F

class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = F.dropout(x, self.p, self.inplace)

        return self.out

    def parameters(self) -> List:
        return []

    def __repr__(self) -> str:
        return f"Dropout(p={self.p}, inplace={self.inplace})"

