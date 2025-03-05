"""
Utility functions for computing classification metrics.
"""

import numpy as np

from uncertainty_toolbox.utils import assert_is_flat_same_shape


def bin_points(scores, bin_edges):
    assert bin_edges is not None, "Bins have not been defined"
    scores = scores.squeeze()
    assert np.size(scores.shape) < 2, "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)


def get_binned_probabilities_fixed_width(y, pred_prob, n_bins, pred_prob_base=None):
    assert n_bins >= 0
    bin_edges = np.linspace(1.0 / n_bins, 1.0, n_bins)
    pi_pred = np.zeros(n_bins)
    pi_base = np.zeros(n_bins)
    pi_true = np.zeros(n_bins)
    n_elem = np.zeros(n_bins)
    bin_assignment = bin_points(pred_prob, bin_edges)

    for i in range(n_bins):
        bin_idx = bin_assignment == i
        n_elem[i] = sum(bin_idx)
        if n_elem[i] == 0:
            continue
        pi_pred[i] = pred_prob[bin_idx].mean()
        if pred_prob_base is not None:
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()

    assert sum(n_elem) == y.size

    return n_elem, pi_pred, pi_base, pi_true


def organize_bin_probabilities(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int,
):
    """Organize predicted probabilities into bins and compute observed probabilities.

    Args:
        y_true: 1D array of observed binary events, of shape (T,).
        y_pred: 1D array of predicted probability of positive events, of shape (T,).
        num_bins: number of bins to use.

    Returns:
        5-tuple of
        - mean probability of each bin.
        - observed probability of the positive class per bin.
        - number of predictions per bin.
        - mean of the predicted probabilities falling into each bin.
        - parity calibration plot.
    """
    assert_is_flat_same_shape(y_true, y_pred)
    num_obs = y_true.shape[0]

    bin_bounds = np.linspace(0, 1, num_bins + 1)
    bins = [(bin_bounds[i], bin_bounds[i + 1]) for i in range(len(bin_bounds) - 1)]
    # midpoint probability of bin
    bin_probs = np.array([np.mean(x) for x in bins])

    obs_prob_list = []
    obs_count_list = []
    hist_obs_list = []
    mean_pred_prob_list = []
    for b in bins:
        l, u = b
        pt_idx = np.logical_and(l <= y_pred, y_pred < u)
        assert pt_idx.shape[0] == num_obs
        if np.sum(pt_idx.astype(int)) == 0:
            # empty bin
            empty_bin_obs_prob = np.nan  # TODO: best way to handle an empty bin?
            empty_bin_mean_pred_prob_in_bin = np.nan
            empty_bin_obs_count = 0
            empty_bin_hist_obs = np.array([])
            obs_prob_list.append(empty_bin_obs_prob)
            obs_count_list.append(empty_bin_obs_count)
            hist_obs_list.append(empty_bin_hist_obs)
            mean_pred_prob_list.append(empty_bin_mean_pred_prob_in_bin)
        else:
            # TODO: can remove the following two assertions
            assert all(l <= y_pred[pt_idx])
            assert all(y_pred[pt_idx] < u)
            # observed probability of positive event in bin
            obs_prob = np.mean(y_true[pt_idx])
            obs_prob_list.append(obs_prob)
            # mean predicted probability in bin
            mean_pred_prob_list.append(np.mean(y_pred[pt_idx]))
            # number of observations in bin
            obs_count_list.append(np.sum(pt_idx.astype(int)))
            # predicted probabilities in bin  # TODO: can remove
            hist_obs_list.append(y_pred[pt_idx])
    mean_pred_probs = np.array(mean_pred_prob_list)
    obs_probs = np.array(obs_prob_list)
    obs_counts = np.array(obs_count_list)
    hist_obs = np.concatenate(hist_obs_list)

    info = {
        "obs_counts": obs_counts,
        "bin_bounds": bin_bounds,
        "hist_obs": hist_obs,
        "bin_probs": bin_probs,
    }
    # return mean_pred_prob, obs_prob, obs_count, hist_obs, bin_probs
    return mean_pred_probs, obs_probs, info
