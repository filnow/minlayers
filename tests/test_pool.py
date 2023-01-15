import unittest
import torch
import nn
import torch.nn.functional as F


class TestMaxPool2d(unittest.TestCase):
    def test_maxpool(self):
        x = torch.randn(1, 3, 224, 224)
        pool = nn.MaxPool2d(kernel_size=3)
        torch.allclose(F.max_pool2d(x, pool.kernel_size, pool.stride, pool.padding), pool(x))

    def test_maxpool_stride(self):
        x = torch.randn(1, 3, 224, 224)
        pool = nn.MaxPool2d(kernel_size=3, stride=2)
        torch.allclose(F.max_pool2d(x, pool.kernel_size, pool.stride, pool.padding), pool(x))

    def test_maxpool_padding(self):
        x = torch.randn(1, 3, 224, 224)
        pool = nn.MaxPool2d(kernel_size=3, padding=1)
        torch.allclose(F.max_pool2d(x, pool.kernel_size, pool.stride, pool.padding), pool(x))

    def test_maxpool_dilation(self):
        x = torch.randn(1, 3, 224, 224)
        pool = nn.MaxPool2d(kernel_size=3, dilation=2)
        torch.allclose(F.max_pool2d(x, pool.kernel_size, pool.stride, pool.padding, pool.dilation), pool(x))


if __name__ == '__main__':
    torch.manual_seed(1337)
    unittest.main()