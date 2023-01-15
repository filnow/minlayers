import torch
from .containers import Module
from typing import Tuple, Optional, Union


class _MaxPool(Module):
    r"""
    Base class for pooling

    Args:
        - kernel_size (int or tuple): the size of the window to take a max over
        - stride (int or tuple, optional): the stride of the window. Default value is kernel_size
        - padding (int or tuple, optional): implicit zero padding to be added on both sides
        - dilation (int or tuple, optional): a parameter that controls the stride of elements in the window
        - ceil_mode (bool, optional): when True, will use `ceil` instead of `floor` to compute the output shape
        
    """
    def __init__(self, 
                 kernel_size: Union[int, Tuple[int, ...]], 
                 stride:  Optional[Union[int, Tuple[int, ...]]] = None, 
                 padding: Union[int, Tuple[int, ...]] = 0, 
                 dilation: Union[int, Tuple[int, ...]] = 1, 
                 ceil_mode: bool = False) -> None:
        self.kernel_size = kernel_size
        self.stride =  stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode


class MaxPool1d(_MaxPool):
    r"""
    1D max pooling layer
    """
    def __init__(self, 
                 kernel_size: Union[int, Tuple[int, ...]], 
                 stride:  Optional[Union[int, Tuple[int, ...]]] = None, 
                 padding: Union[int, Tuple[int, ...]] = 0, 
                 dilation: Union[int, Tuple[int, ...]] = 1, 
                 ceil_mode: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.max_pool1d(x, self.kernel_size, self.stride, 
                                    self.padding, self.dilation, self.ceil_mode)
        
        return self.out
    

class MaxPool2d(_MaxPool):
    r"""
    2D max pooling layer
    """
    def __init__(self, 
                 kernel_size: Union[int, Tuple[int, ...]], 
                 stride:  Optional[Union[int, Tuple[int, ...]]] = None, 
                 padding: Union[int, Tuple[int, ...]] = 0, 
                 dilation: Union[int, Tuple[int, ...]] = 1, 
                 ceil_mode: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.max_pool2d(x, self.kernel_size, self.stride, 
                                    self.padding, self.dilation, self.ceil_mode)
        
        return self.out

