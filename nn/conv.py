import torch
from .containers import Module
from typing import List, Tuple, Optional, Union
import torch.nn.functional as F
#TODO write own implementation of conv1d and conv2d without calling torch.conv1d and torch.conv2d
#TODO change type of kernel size to expect a tuple of ints
#TODO make a file for complicated types like Union[int, Tuple[int, ...]] and import them

class _Conv(Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride:  Union[int, Tuple[int, ...]] = 1, 
                 padding: Union[int, Tuple[int, ...]] = 0, 
                 dilation: Union[int, Tuple[int, ...]] = 1, 
                 groups: int = 1, 
                 bias: bool = True, 
                 padding_mode: str = 'zeros') -> None:

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight: torch.Tensor = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        self.bias: torch.Tensor = torch.randn(out_channels) if bias else torch.zeros(out_channels)
    
    def _conv2d(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.shape

        FILTR, _, HH, WW = self.weight.shape

        HOUT = int(1 + (H + 2 * self.padding - HH) / self.stride)
        WOUT = int(1 + (W + 2 * self.padding - WW) / self.stride)

        x_pad = F.pad(x, (0,0,self.padding,self.padding), "constant", 0)
        out = torch.zeros((N, FILTR, HOUT, WOUT))

        for n in range(N): 
            for f in range(FILTR):  
                for i in range(HOUT): 
                    for j in range(WOUT): 
                        
                        window = x_pad[n, :, i*self.stride:i*self.stride+HH, j*self.stride:j*self.stride+WW]
                        out[n, f, i, j] = torch.sum(window * self.weight[f]) + self.bias[f]
        return out

    def parameters(self) -> List:
        return [self.weight, self.bias] if self.bias else [self.weight]


class Conv1d(_Conv):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride:  Union[int, Tuple[int, ...]] = 1, 
                 padding: Union[int, Tuple[int, ...]] = 0, 
                 dilation: Union[int, Tuple[int, ...]] = 1, 
                 groups: int = 1, 
                 bias: bool = True, 
                 padding_mode: str = 'zeros') -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias, padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.conv1d(x, self.weight, self.bias, 
                                self.stride, self.padding, 
                                self.dilation, self.groups)
        
        return self.out


class Conv2d(_Conv):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride:  Union[int, Tuple[int, ...]] = 1, 
                 padding: Union[int, Tuple[int, ...]] = 0, 
                 dilation: Union[int, Tuple[int, ...]] = 1, 
                 groups: int = 1, 
                 bias: bool = True, 
                 padding_mode: str = 'zeros') -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias, padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = self._conv2d(x)
        
        return self.out



