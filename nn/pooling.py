import torch
from .containers import Module
from typing import Tuple, Optional, Union


class _MaxPool(Module):
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

class AdaptiveAvgPool2d(Module):
    def __init__(self, 
                 output_size: Union[int, Tuple[int, ...]]) -> None:
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)
        
        return self.out