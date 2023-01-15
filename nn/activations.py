import torch
from .containers import Module
from typing import Optional


#TODO : Implement inplace versions of the activation functions

class Tanh(Module):
    r""" 

    Tanh activation function

    Args:
        - x (torch.Tensor): input to the activation function
    
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)
        
        return self.out
    

class Sigmoid(Module):
    r""" 

    Sigmoid activation function

    Args:
        - x (torch.Tensor): input to the activation function
    
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = (1 + torch.exp(-x))**-1
        
        return self.out
    

class ReLU(Module):
    r""" 

    ReLU activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - in_place (bool): if ``True``, will do this operation in-place. Default: ``False
    
    """ 
    def __init__(self, in_place: bool = False) -> None:
        self.in_place = in_place

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.relu(x)
        
        return self.out
    

class LeakyReLU(Module):
    r""" 
    
    LeakyReLU activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - negative_slope (float): controls the angle of the negative slope. Default: 0.01
        - in_place (bool): if ``True``, will do this operation in-place. Default: ``False
    
    """ 
    def __init__(self, negative_slope: float = 0.1, in_place: bool = False) -> None:
        self.negative_slope = negative_slope
        self.in_place = in_place

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x >= 0, x, torch.mul(x, self.negative_slope))
        
        return self.out


class GELU(Module):
    r""" 

    GELU activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - approximate (str): Approximate method to use. Options: 'none', 'tanh'

    Notes:
        - For approximate methods, see: `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
    
    """
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


class Softmax(Module):
    r"""

    GELU activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - dim (int): A dimension along which Softmax will be computed     
    """
    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.softmax(x, dim=self.dim)
        
        return self.out


class ReLU6(Module):
    r"""

    ReLU6 activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - in_place (bool): if ``True``, will do this operation in-place. Default: ``False
    
    Notes:
        - See: `MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications <https://arxiv.org/abs/1704.04861>`_
    
    """ 
    def __init__(self, in_place: bool = False) -> None:
        self.in_place = in_place

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.clamp(x, min=0, max=6)
        
        return self.out


class ELU(Module):
    r"""

    ELU activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - alpha (float): the :math:`\alpha` value for the ELU formulation. Default: 1.0
        - inplace (bool): if ``True``, will do this operation in-place. Default: ``False
    
    Notes:
        - See: `Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) <https://arxiv.org/abs/1511.07289>`_

    """
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        self.alpha = alpha
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x > 0, x, self.alpha*(torch.exp(x) - 1))

        return self.out
    

class Swish(Module):
    r"""

    Swish activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - inplace (bool): if ``True``, will do this operation in-place. Default: ``False
    
    Notes:
        - See: `Searching for Activation Functions <https://arxiv.org/abs/1710.05941>`_

    """
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x * torch.sigmoid(x)
        
        return self.out


class Softplus(Module):
    r"""

    SoftPlus activation function

    Args:
        - x (torch.Tensor): input to the activation function
    
    Notes:
        - See: `Deep Neural Networks for Acoustic Modeling in Speech Recognition: The Shared Views of Four Research Groups <https://arxiv.org/abs/1412.5567>`_

    """
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_input = self.beta * x
        rhs = torch.true_divide(torch.log(1 + torch.exp(scaled_input)), self.beta)

        self.out = torch.where(scaled_input > self.threshold, x, rhs)

        return self.out


class Mish(Module):
    r"""

    Mish activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - inplace (bool): if ``True``, will do this operation in-place. Default: ``False

    Notes:
        - See: `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    """
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace
        self.softplus = Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x * torch.tanh(self.softplus(x))
        
        return self.out


class HardShrink(Module):
    r"""

    HardShrink activation function

    Args:
        - x (torch.Tensor): input to the activation function
        - lambd (float): the :math:`\lambda` value for the HardShrink formulation. Default: 0.5
    
    Notes:
        - See: `Exact solutions to the nonlinear dynamics of learning in deep linear neural networks <https://arxiv.org/abs/1312.6120>`_

    """
    def __init__(self, lambd: float = 0.5) -> None:
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.where(x < -self.lambd, x, torch.where(x > self.lambd, x, torch.zeros_like(x)))

        return self.out