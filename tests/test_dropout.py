import unittest
import torch
import nn
import torch.nn.functional as F


class TestDropout(unittest.TestCase):
    def test_dropout(self):
        x = torch.randn(100, 100)
        dropout = nn.Dropout(0.5)
        torch.allclose(F.dropout(x, dropout.p), dropout(x))

    def test_dropout_no_bias(self):
        x = torch.randn(100, 100)
        dropout = nn.Dropout(0.5, inplace=True)
        torch.allclose(F.dropout(x, dropout.p, dropout.inplace), dropout(x))


if __name__ == '__main__':
    torch.manual_seed(1337)
    unittest.main()