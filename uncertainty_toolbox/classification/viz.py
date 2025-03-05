import numpy as np
import matplotlib.pyplot as plt


from uncertainty_toolbox.classification.metrics_calibration import (
    confidence_expected_calibration_error,
)


def reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int,
    display: bool = True,
    title: str = None,
):
    """Plot parity calibration plot.

    Args:
        obs_fall: array of observed fall events, of shape (T,).
        prob_fall: array of predicted probability of fall events, of shape (T,).
        num_bins: number of bins to use for the parity calibration metrics.
        display: whether to display the plot.
        title: title of the plot.

    Returns:
        # TODO: clean up
        5-tuple of
        - mean probability of each bin.
        - observed probability of the positive class per bin.
        - number of predictions per bin.
        - mean of the predicted probabilities falling into each bin.
        - parity calibration plot.
    """
    conf_ece, info = confidence_expected_calibration_error(
        y_true=y_true, y_pred=y_pred, num_bins=num_bins, return_info=True
    )
    mean_pred_probs = info["mean_pred_probs"]
    obs_probs = info["obs_probs"]
    hist_obs = info["hist_obs"]
    bin_bounds = info["bin_bounds"]
    obs_counts = info["obs_counts"]
    bin_probs = info["bin_probs"]

    # Plot the figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred_probs, obs_probs, "-o")
    ax.plot([0, 1], [0, 1], c="k")
    ax2 = ax.twinx()

    ax2.hist(hist_obs, bins=bin_bounds, alpha=0.3)

    min_bin_count = np.min(obs_counts)

    ax2.set_ylabel("points per bin")

    if title:
        ax.set_title(f"{title} (ece {conf_ece:.6f}, {min_bin_count} min count)")
    else:
        ax.set_title(f"(ece {conf_ece:.6f}, {min_bin_count} min count)")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed probability")

    if display:
        plt.show()

    # TODO: clean up?
    out = {
        "bin_probs": bin_probs,
        "obs_probs": obs_probs,
        "obs_counts": obs_counts,
        "mean_pred_probs": mean_pred_probs,
        "fig": fig,
    }

    return out
