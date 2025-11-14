import numpy as np


def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def weighted_r2_score(y_true, y_pred, weights_dict, target_order):
    w = np.array([weights_dict[k] for k in target_order], dtype=np.float32)

    y_mean = np.mean(y_true, axis=0)
    y_wmean = np.sum(w * y_mean) / np.sum(w)

    ss_res = np.sum(w * np.sum((y_true - y_pred) ** 2, axis=0))
    ss_tot = np.sum(w * np.sum((y_true - y_wmean) ** 2, axis=0))

    return 1.0 - ss_res / ss_tot
