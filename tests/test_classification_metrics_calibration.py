"""
Tests for classification calibration metrics.
"""

import numpy as np
import pytest

from uncertainty_toolbox.classification.metrics_calibration import (
    confidence_expected_calibration_error,
    compute_l1_expected_calibration_error,
    bias_adjusted_sharpness,
    # TODO: Add more function imports here
)


@pytest.fixture
def supply_full_bin_test_set():
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1])
    num_bins = 4
    return y_pred, y_true, num_bins


# TODO: technically, this should not be necessary since this should be tested
# when testing the binning function of classification/utils.py
@pytest.fixture
def supply_empty_bin_test_set():
    y_pred = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y_true = np.array([1, 0, 0, 1, 1, 0, 1])
    num_bins = 4
    return y_pred, y_true, num_bins


@pytest.fixture
def supply_multiclass_test_set():
    # 3-class example with 6 samples
    y_pred = np.array(
        [
            [0.7, 0.2, 0.1],  # predicted class 0
            [0.3, 0.6, 0.1],  # predicted class 1
            [0.1, 0.2, 0.7],  # predicted class 2
            [0.8, 0.1, 0.1],  # predicted class 0
            [0.2, 0.7, 0.1],  # predicted class 1
            [0.1, 0.1, 0.8],  # predicted class 2
        ]
    )
    # Predicted classes:
    # [0, 1, 2, 0, 1, 2]
    y_true = np.array([0, 1, 2, 1, 1, 2])  # true labels
    num_bins = 3
    return y_pred, y_true, num_bins


def test_confidence_expected_calibration_error_on_full_bin_test_set_binary(
    supply_full_bin_test_set,
):
    """Test compute_l1_expected_calibration_error for some dummy values."""
    y_pred, y_true, num_bins = supply_full_bin_test_set
    expected_conf_ece = float(19 / 90)
    actual_conf_ece, info = confidence_expected_calibration_error(
        y_true=y_true,
        y_pred=y_pred,
        num_bins=num_bins,
        return_info=True,
    )
    assert np.abs(expected_conf_ece - actual_conf_ece) < 1e-6


# TODO: technically, this should not be necessary since this should be tested
# when testing the binning function of classification/utils.py
def test_confidence_expected_calibration_error_on_empty_bin_test_set_binary(
    supply_empty_bin_test_set,
):
    """Test compute_l1_expected_calibration_error for some dummy values."""
    y_pred, y_true, num_bins = supply_empty_bin_test_set
    expected_conf_ece = float(6 / 35)
    actual_conf_ece, info = confidence_expected_calibration_error(
        y_true=y_true,
        y_pred=y_pred,
        num_bins=num_bins,
        return_info=True,
    )
    assert np.abs(expected_conf_ece - actual_conf_ece) < 1e-6


def test_confidence_expected_calibration_error_multiclass(supply_multiclass_test_set):
    """Test confidence_expected_calibration_error_multiclass for a small example."""
    y_pred, y_true, num_bins = supply_multiclass_test_set

    actual_conf_ece, info = confidence_expected_calibration_error(
        y_true=y_true,
        y_pred=y_pred,
        num_bins=num_bins,
        return_info=True,
    )

    # For this example:
    # Confidences: [0.7, 0.6, 0.7, 0.8, 0.7, 0.8]
    # Correctness: [1, 1, 1, 0, 1, 1]
    # Expected calibration error ≈ 0.11666666666666671
    expected_conf_ece = 0.11666666666666671

    assert np.abs(expected_conf_ece - actual_conf_ece) < 1e-3


# TODO: place in a separate file?
def test_reliability_diagram(supply_multiclass_test_set):
    import uncertainty_toolbox.classification.viz as viz
    y_pred, y_true, num_bins = supply_multiclass_test_set
    
    viz.reliability_diagram(y_true, y_pred, num_bins, display=True, title="Reliability Diagram")
