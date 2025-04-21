
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

def plot_distributions(stream: tuple, distribution_params: list[tuple], distributions: list[np.ndarray], lengths: list[int]):
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, width_ratios=[3, 1])
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
    ax1.grid()

    data, labels = stream

    ax1.plot(data, label='Stream')
    ax1.set_title('Synthetic Data Stream with Concept Drift')
    ax1.set_xlabel('Sample index')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    transitions = np.cumsum(lengths)
    for t in transitions:
        ax1.axvline(t, color='r', linestyle='--', linewidth=1)

    # Plot histograms of distributions
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    for i, (mean, std) in enumerate(distribution_params):
        # Generate samples from this distribution for histogram
        samples = np.random.normal(mean, std, 1000)
        ax2.hist(samples, alpha=0.5, label=f'Dist {i%5+1}', color=colors[i % 5])

    ax2.set_title('Distributions')
    ax2.grid(axis='y')

    # Add legends
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.show()
