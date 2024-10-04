import matplotlib.pyplot as plt
import numpy as np


def format(ax, xlabel = 'x', ylabel = 'y', title = ''):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax

def coordinate_axes(ax, lw = 1):
    ax.axhline(0, color = 'black', lw = lw, zorder = 3)
    ax.axvline(0, color = 'black', lw = lw, zorder = 3)
    return ax

