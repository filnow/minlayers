import torch
from typing import List


class Module:
  
  def forward(self) -> torch.Tensor:
    raise NotImplementedError
  
  def __call__(self, *args) -> torch.Tensor:
      return self.forward(*args)

  def parameters(self) -> List:
    return []

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}()"


class Sequential(Module):
  
  def __init__(self, *args) -> None:
    self.layers = args
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    
    return x
  
  def parameters(self) -> List:
    return [p for layer in self.layers for p in layer.parameters()]

  def __repr__(self) -> str:
    return f"Sequential({self.layers})"