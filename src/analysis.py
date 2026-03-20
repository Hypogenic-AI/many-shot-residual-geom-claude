"""
Analyze residual stream geometry: dimensionality, probes, belief state regression.
"""
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy import stats


def participation_ratio(activations):
    """Compute participation ratio (effective dimensionality) of activations.

    PR = (sum(lambda_i))^2 / sum(lambda_i^2)
    where lambda_i are eigenvalues of the covariance matrix.

    Args:
        activations: (N, d) array

    Returns:
        float: participation ratio (1 = 1D, d = uniform spread)
    """
    if len(activations) < 2:
        return 1.0
    centered = activations - activations.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim == 0:
        return 1.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)  # numerical stability
    sum_eig = eigvals.sum()
    if sum_eig < 1e-12:
        return 1.0
    return (sum_eig ** 2) / (eigvals ** 2).sum()


def pca_explained_variance(activations, n_components=None):
    """Compute cumulative explained variance ratios.

    Args:
        activations: (N, d) array

    Returns:
        explained_variance_ratio: array of cumulative explained variance
    """
    if n_components is None:
        n_components = min(activations.shape[0], activations.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(activations)
    return np.cumsum(pca.explained_variance_ratio_)


def linear_probe_accuracy(activations, labels, n_folds=5):
    """Train linear probe to predict labels from activations.

    Args:
        activations: (N, d) array
        labels: (N,) int array

    Returns:
        mean accuracy, std accuracy
    """
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return 1.0, 0.0
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    scores = cross_val_score(clf, activations, labels, cv=n_folds, scoring='accuracy')
    return scores.mean(), scores.std()


def belief_state_regression_r2(activations, belief_states):
    """Regress activations onto belief states, return R².

    Args:
        activations: (N, d) array
        belief_states: (N, n_states) array

    Returns:
        R² score
    """
    reg = LinearRegression()
    reg.fit(activations, belief_states)
    return reg.score(activations, belief_states)


def pairwise_distance_correlation(activations, belief_states):
    """Compute R² between pairwise distances in activation space vs belief state space.

    This tests whether the geometric structure is preserved (isometry).

    Args:
        activations: (N, d) array
        belief_states: (N, n_states) array

    Returns:
        R² of pairwise distance correlation
    """
    from scipy.spatial.distance import pdist
    n = min(len(activations), 2000)  # limit for computational feasibility
    if n < len(activations):
        idx = np.random.choice(len(activations), n, replace=False)
        activations = activations[idx]
        belief_states = belief_states[idx]

    d_act = pdist(activations)
    d_bel = pdist(belief_states)
    if len(d_act) < 2:
        return 0.0
    r, _ = stats.pearsonr(d_act, d_bel)
    return r ** 2


def analyze_geometry_by_position(streams, layer, positions_to_analyze=None):
    """Analyze residual stream geometry at different context positions.

    Args:
        streams: dict mapping layer_idx -> (N, T, d) activations
        layer: which layer to analyze
        positions_to_analyze: list of positions, or None for all

    Returns:
        dict with position -> {participation_ratio, pca_var_3, pca_var_5}
    """
    acts = streams[layer]  # (N, T, d)
    if isinstance(acts, torch.Tensor):
        acts = acts.numpy()
    N, T, d = acts.shape

    if positions_to_analyze is None:
        # Sample positions: first, then every 10, then last
        positions_to_analyze = sorted(set([0, 1, 2, 5, 10, 20, 50] +
                                          list(range(0, T, max(1, T // 20))) +
                                          [T - 1]))
        positions_to_analyze = [p for p in positions_to_analyze if p < T]

    results = {}
    for pos in positions_to_analyze:
        a = acts[:, pos, :]  # (N, d)
        pr = participation_ratio(a)
        pca_var = pca_explained_variance(a, n_components=min(10, d, N))
        results[pos] = {
            "participation_ratio": pr,
            "pca_cumvar": pca_var.tolist(),
            "pca_var_3": float(pca_var[min(2, len(pca_var) - 1)]),
            "pca_var_5": float(pca_var[min(4, len(pca_var) - 1)]),
        }
    return results


def analyze_probe_by_position(streams, layer, labels_by_position, positions=None):
    """Probe accuracy at different context positions.

    Args:
        streams: dict mapping layer_idx -> (N, T, d)
        layer: which layer
        labels_by_position: (N, T) int array of labels at each position
        positions: which positions to probe

    Returns:
        dict with position -> (accuracy, std)
    """
    acts = streams[layer]
    if isinstance(acts, torch.Tensor):
        acts = acts.numpy()
    N, T, d = acts.shape

    if positions is None:
        positions = sorted(set([0, 1, 2, 5, 10, 20, 50] +
                               list(range(0, T, max(1, T // 20))) +
                               [T - 1]))
        positions = [p for p in positions if p < T]

    results = {}
    for pos in positions:
        a = acts[:, pos, :]
        labels = labels_by_position[:, pos]
        if len(np.unique(labels)) < 2:
            results[pos] = (1.0, 0.0)
            continue
        acc, std = linear_probe_accuracy(a, labels)
        results[pos] = (acc, std)
    return results
