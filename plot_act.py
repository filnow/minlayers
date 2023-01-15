import nn
import matplotlib.pyplot as plt
import torch
import sys


"""
Tiny program to plot activation functions

Usage:

    - python3 plot_act.py [activation function] - plot a single activation function
    
    - python3 plot_act.py all - plot all activation functions

    - python3 plot_act.py help - print list of activation functions

"""
def plot_act(act, x):

    plt.plot(x, act(x).detach().numpy())
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.title(act.__class__.__name__)
    plt.show()

def plot_all():

    act_list = [nn.Tanh(), 
                nn.Sigmoid(), 
                nn.ReLU(), 
                nn.LeakyReLU(), 
                nn.GELU(), 
                nn.Softmax(dim=0), 
                nn.ReLU6(), 
                nn.ELU(), 
                nn.Swish(), 
                nn.Softplus(), 
                nn.Mish(),
                nn.HardShrink()
                ]

    fig, axs = plt.subplots(3, 4, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle('Activation Functions')
    fig.tight_layout(pad=3.0)
    
    for i, act in enumerate(act_list):

        x = torch.linspace(-7, 7, 100)
        y = act(x).detach().numpy()

        row = i // 4
        col = i % 4

        if row == 2:
            axs[row, col].set_xlabel('Input')

        if col == 0:
            axs[row, col].set_ylabel('Output')
        
        axs[row, col].plot(x, y)
        axs[row, col].grid(linestyle='--', linewidth=1, alpha=0.5)
        axs[row, col].set_xlim(-7, 7)
        axs[row, col].set_ylim(-7, 7)
        axs[row, col].set_title(act.__class__.__name__)

    plt.show()


if __name__ == '__main__':


    if len(sys.argv) < 2:
        print("Usage: python plot_act.py <activation>")
        sys.exit(1)

    act = sys.argv[1]
    
    if act == 'help':
        print("Available activations: Tanh, Sigmoid, ReLU, LeakyReLU, GELU, Softmax, ReLU6, ELU, Swish, Softplus, Mish, all")
        sys.exit(0)

    if act in dir(nn):
        act = getattr(nn, act)()
        plot_act(act, torch.linspace(-7, 7, 100))
    
    elif act == 'all':
        plot_all()

    else:
        print("Activation not found. Use 'help' to see available activations")
        sys.exit(1)

    
