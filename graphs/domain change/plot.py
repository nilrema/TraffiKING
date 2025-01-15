import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random


def color_mesh(data):
    color_map = cm._colormaps['Set3']
    colors = color_map(np.linspace(0, 1, len(data)))
    return colors



def plot_domain_change(data, title, xlabel, ylabel, filename):
    # one barchart with all models
    plt.figure(figsize=(8, 5))
    plt.grid(axis='y', alpha=0.5, zorder=0)
    plt.bar(data.keys(), data.values(), color=color_mesh(data), width=0.5, zorder=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.yticks(np.arange(0, 101, 10))
    plt.tight_layout()
    plt.savefig(f'graphs/domain change/{filename}')


if __name__ == '__main__':
    data = {'TraffiKING': 64.32,
            'ResNet18': 90.23,
            'EfficientNetB0': 83.64,
            'MobileNetV2': 84.51,
            'DenseNet121': 93.25
    }
    plot_domain_change(data, title='Domain Change', xlabel='Model', ylabel='Accuracy', filename='domain_change.png')