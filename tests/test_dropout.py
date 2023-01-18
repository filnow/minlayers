import unittest
import torch
import nn
import torch.nn.functional as F


class TestDropout(unittest.TestCase):
    def test_dropout(self):
        x = torch.randn(100, 100)
        dropout = nn.Dropout()
        torch.testing.assert_close(F.dropout(x), dropout(x))

if __name__ == '__main__':
    torch.manual_seed(1337)
    unittest.main()