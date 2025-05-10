import torch


def z_score(x, mean, std):
    """
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: torch array, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: torch array, z-score normalized array.
    """
    return (x - mean) / std


def un_z_score(x_normed, mean, std):
    """
    Undo the Z-score calculation
    :param x_normed: torch array, input array to be un-normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    """
    return x_normed * std + mean


def WMAPE(v, v_, threshold=0.01):
    """
    Filtered Weighted Mean Absolute Percentage Error.
    Ignores values where ground truth is below a threshold.
    :param v: torch tensor, ground truth.
    :param v_: torch tensor, prediction.
    :param threshold: scalar, minimum value of ground truth to consider.
    :return: torch scalar, WMAPE (%), averaged over filtered values.
    """
    mask = torch.abs(v) >= threshold
    if torch.sum(mask) == 0:
        return torch.tensor(float("nan"))  # or 0, or raise an exception
    return torch.sum(torch.abs(v_ - v)[mask]) / torch.mean(torch.abs(v)[mask]) * 100


def WindowedWMAPE(
    v, v_, ground_truth_unfolded, start_idx, window_radius=1, threshold=0.01
):
    errors = []
    weights = []
    T = ground_truth.shape[0]

    for t in range(len(v)):  # For each of the 9 forecast steps
        pred_val = v_[t]
        min_error = float("inf")
        best_weight = None

        # Define window bounds (stay within ground_truth bounds)
        window_start = max(0, start_idx + t - window_radius)
        window_end = min(T, start_idx + t + window_radius + 1)

        for i in range(window_start, window_end):
            gt_val = ground_truth[i]
            if torch.abs(gt_val) < threshold:
                continue
            abs_error = torch.abs(pred_val - gt_val)
            if abs_error < min_error:
                min_error = abs_error
                best_weight = torch.abs(gt_val)

        if best_weight is not None:
            errors.append(min_error)
            weights.append(best_weight)

    if len(weights) == 0:
        return torch.tensor(float("nan"))

    return torch.sum(torch.stack(errors)) / torch.sum(torch.stack(weights)) * 100


def RMSE(v, v_):
    """
    Root Mean squared error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, RMSE averages on all elements of input.
    """
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MSE(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MSE averages on all elements of input.
    """
    return torch.mean((v_ - v) ** 2)


def MAE(v, v_):
    """
    Mean absolute error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v))


def evaluate_anomalies_with_prediction_tolerance(y_true, y_pred, window=1):
    """
    Evaluate anomaly detection performance by checking if each predicted anomaly
    falls within ±window of any ground truth anomaly.

    Args:
        y_true (array-like): Ground truth binary labels (0 = normal, 1 = anomaly)
        y_pred (array-like): Predicted binary labels (0 = normal, 1 = anomaly)
        window (int): Number of steps on either side within which a prediction is considered correct.

    Returns:
        dict: Precision, Recall, F1 Score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    gt_indices = set(np.where(y_true == 1)[0])
    pred_indices = set(np.where(y_pred == 1)[0])

    matched_gt = set()
    matched_pred = set()

    for pred in pred_indices:
        for gt in gt_indices:
            if abs(pred - gt) <= window and gt not in matched_gt:
                matched_pred.add(pred)
                matched_gt.add(gt)
                break

    TP = len(matched_pred)
    FP = len(pred_indices - matched_pred)
    FN = len(gt_indices - matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "True Positives": TP,
        "False Positives": FP,
        "False Negatives": FN,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
