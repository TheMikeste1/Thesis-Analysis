import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm, uniform


def main():
    # Plot Beta(0.3, 0.3), Beta(3, 0.3), Beta(0.3, 3), Beta(4, 1), Beta(1, 4),
    # Beta(4, 4), each in their own figure
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, 100)
    ax.plot(x * 2 - 1, beta.pdf(x, 0.3, 0.3) * 2 - 1, label="Beta(0.3, 0.3)")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/beta_0.3_0.3.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, 100)
    ax.plot(x * 2 - 1, beta.pdf(x, 3, 0.3) * 2 - 1, label="Beta(3, 0.3)")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/beta_3_0.3.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, 100)
    ax.plot(x * 2 - 1, beta.pdf(x, 0.3, 3) * 2 - 1, label="Beta(0.3, 3)")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/beta_0.3_3.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, 100)
    ax.plot(x * 2 - 1, beta.pdf(x, 4, 1) * 2 - 1, label="Beta(4, 1)")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/beta_4_1.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, 100)
    ax.plot(x * 2 - 1, beta.pdf(x, 1, 4) * 2 - 1, label="Beta(1, 4)")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/beta_1_4.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, 100)
    ax.plot(x * 2 - 1, beta.pdf(x, 4, 4) * 2 - 1, label="Beta(4, 4)")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/beta_4_4.png")
    plt.close(fig)

    # Plot Gaussian
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(-1, 1, 100)
    ax.plot(x, norm.pdf(x, 0, 1), label="Gaussian")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/gaussian.png")
    plt.close(fig)

    # Plot Uniform
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(-1, 1, 100)
    ax.plot(x, uniform.pdf(x, -1, 2), label="Uniform")
    ax.axes.yaxis.set_ticklabels([])
    plt.tick_params(left = False)
    fig.savefig("img/dists/uniform.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
