"""Post-processing only: aggregate association results produced by MetaXcan."""

import numpy as np
from scipy.special import logsumexp


def acat_log_pvalue(log_pvalues):
    """Equal-weight ACAT in log space, omitting undefined draws."""
    values = np.asarray(log_pvalues, dtype=float)
    values = values[~np.isnan(values)]
    if (values > 0).any():
        raise ValueError("Log p-values must be nonpositive")
    if not len(values):
        return float("nan")
    if len(values) == 1 or np.all(values == values[0]):
        return float(values[0])
    if np.isneginf(values).any():
        return -np.inf
    small = values < np.log(1e-15)
    log_terms = np.empty_like(values)
    signs = np.ones_like(values)
    log_terms[small] = -values[small] - np.log(np.pi)
    terms = np.tan(np.pi * (0.5 - np.exp(values[~small])))
    signs[~small] = np.sign(terms)
    with np.errstate(divide="ignore"):
        log_terms[~small] = np.log(np.abs(terms))
    log_statistic, sign = logsumexp(log_terms, b=signs, return_sign=True)
    log_statistic -= np.log(len(values))
    if sign == 0:
        return -np.log(2)
    if log_statistic > 35:
        tail_logp = -log_statistic - np.log(np.pi)
        return float(tail_logp if sign > 0 else np.log1p(-np.exp(tail_logp)))
    if log_statistic < -35:
        return -np.log(2)
    tail = np.arctan(np.exp(-log_statistic)) / np.pi
    return float(np.log(tail) if sign > 0 else np.log1p(-tail))


def benjamini_hochberg_log(log_pvalues):
    """BH-adjusted log p-values, preserving missing tests and extreme tails."""
    values = np.asarray(log_pvalues, dtype=float)
    result = np.full(values.shape, np.nan)
    indices = np.flatnonzero(~np.isnan(values))
    if not len(indices):
        return result
    if (values[indices] > 0).any():
        raise ValueError("Log p-values must be nonpositive")
    order = indices[np.argsort(values[indices], kind="stable")]
    adjusted = values[order] + np.log(len(order)) - np.log(np.arange(1, len(order) + 1))
    result[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 0)
    return result


def summarize_draws(result):
    valid = result["status"] == "tested"
    count = int(valid.sum())
    logp = acat_log_pvalue(result["log_pvalue"][valid])
    summary = {"n_draws_tested": count,
               "status": "tested" if count else "no_testable_draws",
               "pvalue": float(np.exp(logp)), "neg_log10_pvalue": float(-logp / np.log(10)),
               "log_pvalue": logp}
    if count == 1:
        summary["pvalue"] = float(result["pvalue"][valid][0])
    for name in ("zscore", "effect_size"):
        summary[name] = float(np.mean(result[name][valid])) if count else np.nan
    return summary
