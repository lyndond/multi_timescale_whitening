import collections
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
# import scipy as sp
from scipy.linalg import orthogonal_procrustes
import seaborn as sns
from tqdm import tqdm 

# from timescales.frame import normalize_frame
# from timescales.stats import psd_sqrt, normalize_cov


def optimal_alignment(W, W0, align_type='permutation'):
    """finds the optimal permutation of W to match W0, allowing for sign flips
    All columns of W and W0 are assumed to be unit vectors.
    """
    assert align_type in ['permutation', 'rotation']

    R, _ = orthogonal_procrustes(W, W0)  # minimizes ||W@R - W0||_F

    if align_type == 'rotation':
        W_aligned = W@R
    else: # approx perm P by setting the largest abs element to 1 or -1 and the rest to 0
        P = np.zeros_like(R)
        for i in range(R.shape[1]):
            r = R[:, i]

            # set the largest abs element to 1 or -1 and the rest to 0
            idx = np.argmax(np.abs(r))
            p = np.zeros_like(r)
            p[idx] = np.sign(r[idx])
            P[:, i] = p

        W_aligned = W@P
    
    return W_aligned


def context_adaptation_experiment(
        Css_list, 
        N: int, 
        K: int, 
        eta_w: float, 
        n_samples: int, 
        g0=None, 
        W0=None, 
        normalize_w=False,
        seed=None, 
        error_ord=2, 
        verbose=True,
        ):
    rng = np.random.default_rng(seed)
    
    # make sources
    Css12_list = [psd_sqrt(C) for C in Css_list]
    n_contexts = len(Css_list)
    # W_dct = sp.fftpack.dct(np.eye(N), norm='ortho')

    if W0 is None:
        if N == K:
            W, _ = np.linalg.qr(rng.standard_normal((N, N)))
        else:
            W = rng.standard_normal((N, K))
        W = normalize_frame(W) #* 1/np.sqrt(K)
        W0 = W.copy()
    else:
        W = W0.copy()

    In = np.eye(N)
    g_all = []
    error = []
    dW_norm = []
 
    # if n_samples == 0, then run for n_contexts without sampling
    T = n_contexts if n_samples == 0 else n_samples
    iterator = tqdm(range(T)) if verbose else range(T)
    for t in iterator:
        ctx = rng.integers(0, n_contexts, (1, ) )[0] if n_samples > 0 else t
        Css, Css12 = Css_list[ctx], Css12_list[ctx]

        WTW = W.T @ W
        if g0 is None:
            g = np.linalg.solve(WTW**2, np.diag(W.T @ Css12 @ W - WTW))
        else:
            g = g0
        # G = np.diag(g)

        WG = W * g[None, :]
        WGWT = WG @ W.T  # more efficient way of doing W @ np.diag(g) @ W.T
        # M = np.linalg.inv(In + WGWT)
        M = np.linalg.solve(In + WGWT, In)
        Crr = M @ Css @ M.T

        # update W
        # dW = (Crr @ W @ G) - W@G
        dW = (Crr @ WG) - WG
        W = W + eta_w * dW
        dW_norm.append(np.linalg.norm(dW))
        if normalize_w:
            W = normalize_frame(W)
    
        error.append(compute_error(Crr, In, error_ord))
        g_all.append(g)

    results = {
        'W0': W0,
        'W': W,
        'g': np.stack(g_all, 0),
        'error': error,
        'dW_norm': dW_norm,
        'N': N,
        'K': K,
        'eta_w': eta_w,
        'n_samples': n_samples,
        'g0': g0,
        'W0': W0,
        'seed': seed,
    }
    
    return results


def get_cov_controls(Css_list, seed):
    # generate a spectrally-matched set of covariances, with random eigenbasis
    rng = np.random.default_rng(seed)

    Css_ctrl = []
    for C in Css_list:
        d, _ = np.linalg.eigh(C)
        V, _ = np.linalg.qr(rng.standard_normal(C.shape))
        d = np.diag(d)
        Css_ctrl.append(V @ d @ V.T)
    
    return Css_ctrl


def find_peak_frequency_idx(x):
    # find the peak frequency
    X = np.abs(np.fft.fft(x))
    return np.argmax(X)


def sort_cols_by_peak_frequency(X):
    X = X.copy()
    idx = [find_peak_frequency_idx(X[:, i]) for i in range(X.shape[1])]
    return X[:, np.argsort(idx)]


def plot_corrs(filtered_covs, n_rows=3, n_cols=3):

    vmax = 1
    # plot 100 covs
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 10), sharex='all', sharey='all')
    ax = ax.ravel()
    for i, C in enumerate(filtered_covs[:len(ax)]):
        ax[i].imshow(normalize_cov(C), cmap=sns.color_palette('icefire', as_cmap=True), vmin=-vmax, vmax=vmax)
        ax[i].axis('off')
    fig.tight_layout()


# helpers
def normalize_frame(W, axis=0):
    """Normalize the columns of W to unit length"""
    W0 = W / np.linalg.norm(W, axis=axis, keepdims=True)
    return W0


def psd_sqrt(C):
    """Computes PSD square root"""
    d, V = np.linalg.eigh(C)
    D_sqrt = np.diag(np.sqrt(np.abs(d)))  # ensure positive eigenvals
    Csqrt = V @ D_sqrt @ V.T
    return Csqrt


def psd_square(C):
    """Computes PSD square"""
    C_squared = C.T @ C
    return C_squared


def compute_error(C, Ctarget, ord=2):
    return np.linalg.norm(C-Ctarget, ord=ord)


def get_g_opt(
    W,
    Css,
    alpha = 1.0,
):
    """Compute optimal G."""
    N, K = W.shape
    # assert K == N * (N + 1) // 2, "W must have K = N(N+1)/2 columns."
    In = np.eye(N)
    gram_sq_inv = np.linalg.inv((W.T @ W) ** 2)
    Css_12 = psd_sqrt(Css)
    g_opt = gram_sq_inv @ np.diag(W.T @ (Css_12 - alpha*In) @ W)
    return g_opt


def smooth(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, 'valid')


def rot2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def min_dist_2d(V, W):
    # enumerate all possible column axis flips and permutations for 2x2 matrix
    Q0 =[
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, -1]]),
        np.array([[-1, 0], [0, 1]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [-1, 0]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[0, -1], [-1, 0]]),
    ]
    dists = [np.linalg.norm(V - W @ Q) for Q in Q0]
    return W @ Q0[np.argmin(dists)]


def context_adaptation_experiment2(
        Css_list: List[np.ndarray], 
        N: int, 
        K: int, 
        eta_w: float, 
        eta_g: float,
        batch_size: int,
        n_context_samples: int, 
        n_samples: int, 
        g0: np.ndarray,
        W0: np.ndarray, 
        alpha=1.,
        online=True,
        normalize_w=False,
        seed=None, 
        error_ord=2, 
        lock_pre_post=True,
        verbose: bool = True,
        ) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    
    # make sources
    Css12_list = [psd_sqrt(Css) for Css in Css_list]
    n_contexts = len(Css_list)

    N, K = W0.shape
    W = W0.copy()

    In = np.eye(N)

    results = collections.defaultdict(list)

    T = n_context_samples
    iterator = tqdm(range(T)) if verbose else range(T)
    g = g0.copy()

    # prepend and append the same 10 random contexts for plotting later
    n_pre_post = 10
    pre_post = rng.integers(0, n_contexts, (n_pre_post,))
    contexts = np.concatenate([pre_post, rng.integers(0, n_contexts, (T-2*n_pre_post, ) ), pre_post])

    if lock_pre_post:
        # don't adjust W for first and last n_pre_post contexts
        eta_w_list = [0.] * n_pre_post + [eta_w] * (T-2*n_pre_post) + [0.] * n_pre_post  
    else:
        eta_w_list = [eta_w] * T

    presented_contexts = []
    for t in iterator:
        ctx = contexts[t]
        Css, Css12 = Css_list[ctx],  Css12_list[ctx]
        presented_contexts.append(ctx)
        eta_w0 = eta_w_list[t]
        for _ in range(n_samples):

            if online:
                # draw sample and compute primary neuron steady-state
                s = Css12 @ rng.standard_normal((N, batch_size))  # sample data
                WGWT = (W * g[None, :]) @ W.T  # more efficient way of doing W @ np.diag(g) @ W.T
                F = np.linalg.solve(alpha*In + WGWT, In)  # more stable than inv
                r =  F @ s  # primary neuron steady-state

                # compute interneuron input/output and update g
                z = W.T @ r  # interneuron steady-state input
                n = g[:, None] * z   # interneuron steady-state output
                w_norm = np.linalg.norm(W, axis=0)**2  # weight norm
                dg = z**2 - w_norm[:, None]
                g = g + eta_g * np.mean(dg, -1)

                # update W
                rnT = r @ n.T / batch_size
                dW = rnT - W * g[None, :]
                W = W + eta_w0 * dW
                Crr = F @ Css @ F.T

            else:
                WG = W * g[None, :]
                WGWT = WG @ W.T  # more efficient way of doing W @ np.diag(g) @ W.T
                F = np.linalg.solve(alpha*In + WGWT, In)  # more stable than inv
                Crr = F @ Css @ F.T

                # efficient way of computing diag(W.T @ Crr @ W)
                tmp = Crr @ W
                variances = np.array([w@t for w,t in zip(W.T, tmp.T)])

                # update g 
                dg = variances - np.linalg.norm(W, axis=0)**2
                g = g + eta_g * dg
                
                # update W
                dW = (Crr @ WG) - WG
                W = W + eta_w0 * dW

            W = normalize_frame(W) if normalize_w else W
            results['g'].append(g)
            results['g_norm'].append(np.linalg.norm(g))
            results['W_norm'].append(np.linalg.norm(W))
            results['dg_norm'].append(np.linalg.norm(dg))
            results['dW_norm'].append(np.linalg.norm(dW))
            results['error'].append(compute_error(Crr, In, error_ord))

    results.update({
        'W0': W0,
        'W': W,
        'N': N,
        'K': K,
        'eta_w': eta_w,
        'n_samples': n_samples,
        'presented_contexts': presented_contexts,
        'g0': g0,
        'W0': W0,
        'seed': seed,
    })
    return results


def make_contexts(
    N: int,
    L: int,
    n_unique_contexts: int, 
    min_lmbda: float = 0.,
    max_lmbda: float = 10.,
    rng: Optional[np.random.Generator]=None,
    ):
    if rng is None:
        rng = np.random.default_rng()

    V = rng.standard_normal((N, L))

    eps = 1E-1
    if N==L and N ==2:  # sample a non-orthogonal, but well-conditioned basis
        v0 = normalize_frame(rng.standard_normal((N,1)))
        theta_bounds = np.deg2rad(np.array((40, 80)))
        v1 = rot2d(rng.uniform(*theta_bounds)) @ v0
        V = np.concatenate([v0, v1], axis=1)
    elif N == L:
        V, _ =  np.linalg.qr(eps * rng.standard_normal((N, N)))

    V = normalize_frame(V)

    lmbdas = []
    while len(lmbdas) < n_unique_contexts:
        lmbda = rng.uniform(min_lmbda, max_lmbda, size=(L,))
        sparse_mask = rng.uniform(0., 1, size=(L,)) < 0.5   # sparsity mask
        lmbda = lmbda * sparse_mask

        if sum(lmbda) > 0:  # don't append if all zeros
            lmbdas.append(lmbda)

    Sigmas = [np.eye(N) + (V * lmbda[None,:]) @ V.T for lmbda in lmbdas] 
    Css_list = [S.T@S for S in Sigmas]

    return V, Css_list
