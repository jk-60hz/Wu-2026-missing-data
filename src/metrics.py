"""Evaluation metrics, computed only on the evaluated (masked) entries."""
import numpy as np


def paper_metrics(truth, pred):
    """Paper Eq. 19: Pearson rho and RMSE normalised by the std of the ground truth."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    if len(truth) < 3 or np.std(truth) == 0 or np.std(pred) == 0:
        rho = np.nan
    else:
        rho = float(np.corrcoef(truth, pred)[0, 1])
    sd = np.std(truth)
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    return dict(rho=rho, nrmse=rmse / sd if sd > 0 else np.nan)


def solar_metrics(truth, pred, capacity):
    """Practitioner metrics. capacity = nominal max of the series (same units)."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    err = pred - truth
    m = paper_metrics(truth, pred)
    tot = truth.sum()
    m.update(
        mae_cap=float(np.mean(np.abs(err)) / capacity * 100),       # % of capacity
        rmse_cap=float(np.sqrt(np.mean(err ** 2)) / capacity * 100),  # % of capacity
        bias_cap=float(np.mean(err) / capacity * 100),
        energy_err=float(abs(pred.sum() - tot) / tot * 100) if tot > 0 else np.nan,  # % of true energy
        n=int(len(truth)),
    )
    return m
