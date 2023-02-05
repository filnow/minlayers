import numpy as np


def tanh(x): return np.tanh(x)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def relu(x): return np.maximum(0, x)

def leaky_relu(x, negative_slope=0.01): return np.where(x >= 0, x, negative_slope * x)

def softmax(x, dim=0): return np.exp(x) / np.sum(np.exp(x), axis=dim)

def relu6(x): return np.minimum(relu(x), 6)

def gelu(x): return x * 0.5 * (1.0 + np.erf(x / 1.41421))

def swish(x): return x * sigmoid(x)

def elu(x, alpha=1.0): return np.where(x > 0, x, alpha*(np.exp(x) - 1))

