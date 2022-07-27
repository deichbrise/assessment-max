import numpy as np


def rmspe(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Percentage Error (RMSPE).

    As defined in https://www.kaggle.com/competitions/rossmann-store-sales/overview/evaluation. # noqa: E501

    Instances with y==0 are ignored.

    Args:
        y: array-like of shape (n_samples,)
        y_pred: array-like of shape (n_samples,)

    Returns:
        Score: float
    """
    non_zero_sales = y != 0
    y_masked = y[non_zero_sales]
    y_pred_masked = y_pred[non_zero_sales]
    score = np.sqrt(np.mean((y_masked - y_pred_masked) / y_masked) ** 2)
    assert score >= 0.0
    return score
