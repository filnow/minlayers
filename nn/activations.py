import torch
from .containers import Module, Sequential
from .linear import Linear
from .dropout import Dropout
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
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.relu(x)
        
        return self.out
    
    def __repr__(self) -> str:
        return f"ReLU(inplace={self.inplace})"


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x >= 0, x, self.negative_slope * x)
        
        return self.out

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope}, inplace={self.inplace})"


class GELU(Module):
    def __init__(self, approximate: str = 'none') -> None:
        super().__init__()
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
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.softmax(x, dim=self.dim)
        
        return self.out

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class ReLU6(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.clamp(x, min=0, max=6)
        
        return self.out

    def __repr__(self) -> str:
        return f"ReLU6(inplace={self.inplace})"


class ELU(Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x > 0, x, self.alpha*(torch.exp(x) - 1))

        return self.out
    
    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha}, inplace={self.inplace})"


class Swish(Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x * torch.sigmoid(x)
        
        return self.out

    def __repr__(self) -> str:
        return f"Swish(inplace={self.inplace})"


class Softplus(Module):
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super().__init__()
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
        super().__init__()
        self.inplace = inplace
        self.softplus = Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x * torch.tanh(self.softplus(x))
        
        return self.out

    def __repr__(self) -> str:
        return f"Mish(inplace={self.inplace})"


class HardShrink(Module):
    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x < -self.lambd, x, torch.where(x > self.lambd, x, torch.zeros_like(x)))

        return self.out

    def __repr__(self) -> str:
        return f"HardShrink(lambd={self.lambd})"


class Attention(Module):
    def __init__(self, 
                 n_embd: int, 
                 head_size: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.key = Linear(n_embd, head_size, bias=False)
        self.query = Linear(n_embd, head_size, bias=False)
        self.value = Linear(n_embd, head_size,  bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = q.shape
        
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        attn = query @ key.transpose(-2, -1) * C**-0.5
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask[:T, :T] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn = attn @ value

        return attn


class MultiheadAttention(Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int,
                 dropout: int = 0.0) -> None:
        super().__init__()
        head_size = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.heads = [Attention(embed_dim, head_size) for _ in range(num_heads)]
        self.proj = Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.out = torch.cat([head(q, k, v, attn_mask) for head in self.heads], dim=-1)
        self.out = self.dropout(self.proj(self.out))

        return self.out
