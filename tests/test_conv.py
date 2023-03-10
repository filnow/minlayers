import unittest
import torch
import nn
import torch.nn.functional as F

#TODO simplify this test

class TestConv2d(unittest.TestCase):
    def test_conv2d(self):
        x = torch.randn(1, 3, 6, 6)
        conv = nn.Conv2d(3, 32, kernel_size=3)
        torch.testing.assert_close(F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding), conv(x))

    def test_conv2d_no_bias(self):
        x = torch.randn(1, 3, 6, 6)
        conv = nn.Conv2d(3, 32, kernel_size=3, bias=False)
        torch.testing.assert_close(F.conv2d(x, conv.weight, None, conv.stride, conv.padding), conv(x))

    def test_conv2d_stride(self):
        x = torch.randn(1, 3, 6, 6)
        conv = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        torch.testing.assert_close(F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding), conv(x))

    def test_conv2d_padding(self):
        x = torch.randn(1, 3, 6, 6)
        conv = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        torch.testing.assert_close(F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding), conv(x))

if __name__ == '__main__':
    torch.manual_seed(1337)
    unittest.main()