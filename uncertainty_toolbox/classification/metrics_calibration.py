"""
Metrics for assessing quality of predictive uncertainty quantification in classification.

- Confidence Calibration: treat as a binary problem where the true class is the class with highest pred_prob
- Classwise Calibration: treat it as a binary problem for each class separately
"""

import numpy as np

from uncertainty_toolbox.classification.utils import (
    organize_bin_probabilities,
    get_binned_probabilities_fixed_width,
)


def confidence_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int,
    return_info: bool = False,
):
    """Calculate Confidence Expected Calibration Error (a.k.a. Conf-ECE) for Multiclass

    Args:
        y_true: 1D array of true labels, of shape (T,).
        y_pred: 2D array of predicted probability , of shape (T, k).
        num_bins: number of bins to use.
        return_info: whether to return additional information on bins.

    Returns:
        Confidence Expected Calibration Error (Conf-ECE) metric:
            sum of L1 error between mean_predicted_probability and observed probability
            across each bin, weighted by the proportion of predictions falling
            into each bin.

    """
    is_multiclass = y_pred.ndim == 2
    if is_multiclass:
        pred_class = np.argmax(y_pred, axis=1)  # (T,)
        confidence = np.max(y_pred, axis=1)  # (T,)
        assert pred_class.shape == y_true.shape
        correct = (pred_class == y_true).astype(int)  # (T,)
    else:
        assert y_pred.ndim == 1
        assert y_pred.shape == y_true.shape
        correct = y_true
        confidence = y_pred
    mean_pred_probs, obs_probs, info = organize_bin_probabilities(
        y_true=correct, y_pred=confidence, num_bins=num_bins
    )
    obs_counts = info["obs_counts"]
    conf_ece = compute_l1_expected_calibration_error(
        mean_pred_probs=mean_pred_probs,
        obs_probs=obs_probs,
        obs_counts=obs_counts,
    )
    if return_info:
        info["mean_pred_probs"] = mean_pred_probs
        info["obs_probs"] = obs_probs
        return conf_ece, info
    return conf_ece


# TODO: deprecated
def confidence_expected_calibration_error_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int,
    return_info: bool = False,
):
    """Calculate Confidence Expected Calibration Error (a.k.a. Conf-ECE).

    Args:
        y_true: 1D array of observed binary events, of shape (T,).
        y_pred: 1D array of predicted probability of positive events, of shape (T,).
        num_bins: number of bins to use.
        return_info: whether to return additional information on bins.

    Returns:
        Confidence Expected Calibration Error (Conf-ECE) metric:
            sum of L1 error between mean_predicted_probability and observed probability
            across each bin, weighted by the proportion of predictions falling
            into each bin.

    """

    mean_pred_probs, obs_probs, info = organize_bin_probabilities(
        y_true=y_true, y_pred=y_pred, num_bins=num_bins
    )
    obs_counts = info["obs_counts"]
    conf_ece = compute_l1_expected_calibration_error(
        mean_pred_probs=mean_pred_probs,
        obs_probs=obs_probs,
        obs_counts=obs_counts,
    )
    if return_info:
        info["mean_pred_probs"] = mean_pred_probs
        info["obs_probs"] = obs_probs
        return conf_ece, info
    return conf_ece


# TODO: maybe rename this?
def compute_l1_expected_calibration_error(
    mean_pred_probs: np.ndarray,
    obs_probs: np.ndarray,
    obs_counts: np.ndarray,
):
    """Compute the l1 expected calibration error metric.

    Assumes
        - mean predicted probability of each bin
        - observed probability of the positive class per bin
        - number of predictions per bin
    are given.

    Args:
        mean_pred_probs (np.ndarray): mean of the predicted probabilities falling into each bin.
        obs_probs (np.ndarray): the observed probabilities of the positive class per bin.
        obs_counts (np.ndarray): number of predictions per bin.

    Returns:
        float: l1 expected calibration error metric.
    """
    # num_positive_bins = np.sum(obs_counts > 0)
    props_per_bin = obs_counts / np.sum(obs_counts)
    l1_expected_calibration_error = np.nansum(
        props_per_bin * np.abs(mean_pred_probs.flatten() - obs_probs.flatten())
    )

    return l1_expected_calibration_error


def bias_adjusted_sharpness(y, pred_prob, n_bins=15):
    n_elem, _, _, pi_true = get_binned_probabilities_fixed_width(y, pred_prob, n_bins)
    assert sum(n_elem) == y.size
    biased_estimate = (n_elem @ (pi_true**2)) / y.size
    correction = (
        n_elem @ np.divide(np.multiply(pi_true, 1 - pi_true), np.maximum(n_elem - 1, 1))
    ) / y.size
    return biased_estimate - correction  # np.sum(n_elem * (pi_true**2))/y.size
