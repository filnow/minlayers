import numpy as np

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = np.random.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
    self.bias = np.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


