import torch
from typing import List, Union, Tuple
from .containers import Module


class BatchNorm1d(Module):
  def __init__(self, 
               num_features: Union[int, Tuple[int,int]], 
               eps: float = 1e-5, 
               momentum: float = 0.1) -> None:

    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.training: bool = True
    self.gamma: torch.Tensor = torch.ones(num_features)
    self.beta: torch.Tensor = torch.zeros(num_features)
    self.running_mean: torch.Tensor = torch.zeros(num_features)
    self.running_var: torch.Tensor = torch.ones(num_features)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.training:
      if x.ndim == 2:
        self.num_features = 0
      elif x.ndim == 3:
        self.num_features = (0,1)
      xmean = x.mean(self.num_features, keepdim=True) # batch mean
      xvar = x.var(self.num_features, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self) -> List:
    return [self.gamma, self.beta]

  def __repr__(self) -> str:
    return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"


class LayerNorm(Module):
  def __init__(self, 
               num_features: Union[int, Tuple[int,int]], 
               eps: float = 1e-5) -> None:

    self.eps = eps
    self.gamma: torch.Tensor = torch.ones(num_features)
    self.beta: torch.Tensor = torch.zeros(num_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    #NOTE this is different from BatchNorm1d beacuse we are normalizing over the rows not the columns  
    self.num_features = 1 if x.ndim == 2 else (1,2)
    
    xmean = x.mean(self.num_features, keepdim=True) 
    xvar = x.var(self.num_features, keepdim=True) 
    
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) 
    self.out = self.gamma * xhat + self.beta

    return self.out
  
  def parameters(self) -> List:
    return [self.gamma, self.beta]

  def __repr__(self) -> str:
    return f"LayerNorm(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"
