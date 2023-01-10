import numpy as np
from typing import List


class Tanh:     
  def __call__(self, x) -> None:
    self.out = np.tanh(x)
    return self.out
  
  def parameters(self) -> List:
    return []


class Sigmoid:
    def __call__(self, x) -> None:
        self.out = (1 + np.exp(-x))**-1
        return self.out
    
    def parameters(self) -> List:
        return []


class ReLU: 
    def __call__(self, x) -> None:
        self.out = np.maximum(0, x)
        return self.out
    
    def parameters(self) -> List:
        return []


