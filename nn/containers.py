import torch
from typing import List

class Module:
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
  
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
      # Overriding the __call__ function
      return self.forward(x)

  def parameters(self) -> List:
    return []

class Sequential(Module):
  
  def __init__(self, layers) -> None:
    self.layers = layers
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    
    return x
  
  def parameters(self) -> List:
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]