import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_frame2d(
    W: npt.NDArray[np.float64], ax=None, plot_line: bool = False, **kwargs
) -> None:
    """Plots 2D frame vectors, optionally plot the axes along which they lie"""
    assert W.shape[0] == 2, "R must be in R2"
    _, n_vectors = W.shape
    if ax is None:
        ax = plt
    for i in range(n_vectors):
        if i > 0 and "label" in kwargs:
            kwargs.pop("label")
        ax.plot([0, W[0, i]], [0, W[1, i]], "-o", **kwargs)

    x = np.linspace(-2, 2, 10)
    G2 = (W.T @ W) ** 2
    G2 = np.tril(G2, k=-1)
    G2[np.isclose(G2, 0)] = np.inf
    ind = np.unravel_index(G2.argmin(), G2.shape)
    if plot_line:  # plot axis along which the vectors lie
        [
            ax.plot(x, (W[1, i] / W[0, i]) * x, "--", color="r" if i in ind else "k")
            for i in range(W.shape[1])
        ]


def plot_ellipse(
    C: npt.NDArray[np.float64], n_pts: int = 100, ax=None, stdev: float =2., **kwargs
) -> None:
    """Plots 2D 1-stdev ellipse according to covariance matrix C"""
    assert C.shape == (2, 2)
    thetas = np.linspace(0, 2 * np.pi, n_pts)
    dots = np.stack([np.cos(thetas), np.sin(thetas)]) * stdev
    E, V = np.linalg.eigh(C)
    ellipse = V @ np.diag(np.sqrt(E)) @ dots
    if ax is None:
        ax = plt
    ax.plot(ellipse[0, :], ellipse[1, :], **kwargs)
