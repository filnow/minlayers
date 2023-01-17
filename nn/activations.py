import torch
from .containers import Module
from typing import Optional


#TODO : Implement inplace versions of the activation functions


class Tanh(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)
        
        return self.out

    def __repr__(self) -> str:
        return f"Tanh()"
    

class Sigmoid(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = (1 + torch.exp(-x))**-1
        
        return self.out
    
    def __repr__(self) -> str:
        return f"Sigmoid()"


class ReLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.relu(x)
        
        return self.out
    
    def __repr__(self) -> str:
        return f"ReLU(inplace={self.inplace})"


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.1, inplace: bool = False) -> None:
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x >= 0, x, torch.mul(x, self.negative_slope))
        
        return self.out

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope}, inplace={self.inplace})"


class GELU(Module):
    def __init__(self, approximate: str = 'none') -> None:
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.approximate == 'none':
            self.out = x * 0.5 * (1.0 + torch.erf(x / 1.41421))     
        
        elif self.approximate == 'tanh':
            self.out = x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x**2))) 
        
        else:
            raise ValueError('Approximate method not supported')
        
        return self.out

    def __repr__(self) -> str:
        return f"GELU(approximate={self.approximate})"


class Softmax(Module):
    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.softmax(x, dim=self.dim)
        
        return self.out

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class ReLU6(Module):
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.clamp(x, min=0, max=6)
        
        return self.out

    def __repr__(self) -> str:
        return f"ReLU6(inplace={self.inplace})"


class ELU(Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        self.alpha = alpha
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x > 0, x, self.alpha*(torch.exp(x) - 1))

        return self.out
    
    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha}, inplace={self.inplace})"


class Swish(Module):
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x * torch.sigmoid(x)
        
        return self.out

    def __repr__(self) -> str:
        return f"Swish(inplace={self.inplace})"


class Softplus(Module):
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_input = self.beta * x
        rhs = torch.true_divide(torch.log(1 + torch.exp(scaled_input)), self.beta)

        self.out = torch.where(scaled_input > self.threshold, x, rhs)

        return self.out

    def __repr__(self) -> str:
        return f"Softplus(beta={self.beta}, threshold={self.threshold})"


class Mish(Module):
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace
        self.softplus = Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x * torch.tanh(self.softplus(x))
        
        return self.out

    def __repr__(self) -> str:
        return f"Mish(inplace={self.inplace})"


class HardShrink(Module):
    def __init__(self, lambd: float = 0.5) -> None:
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x < -self.lambd, x, torch.where(x > self.lambd, x, torch.zeros_like(x)))

        return self.out

    def __repr__(self) -> str:
        return f"HardShrink(lambd={self.lambd})"

        