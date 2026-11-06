"""Deterministic and member/bootstrap elastic-net students with HWE projection."""

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import ElasticNet, ElasticNetCV


def normal_targets(means, variances=None):
    """Rank within gene/cell type across donors, as in Supplement B.3.2.

    Inputs are already on the teacher's log1p scale. Map standard deviations
    through the interpolated rank CDF before computing inverse-variance weights.
    """
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2 or not np.isfinite(means).all():
        raise ValueError("Expected finite member-by-individual teacher means")
    n = means.shape[1]
    quantiles = rankdata(means, method="average", axis=1) / (n + 1)
    transformed = ndtri(quantiles)
    if variances is None:
        return transformed, None
    variances = np.asarray(variances, dtype=np.float64)
    if variances.shape != means.shape or not np.isfinite(variances).all() or (variances < 0).any():
        raise ValueError("Teacher variances must be finite, nonnegative, and aligned with means")
    mapped_variances = np.empty_like(variances)
    for i, (mean, variance) in enumerate(zip(means, variances)):
        order = np.argsort(mean, kind="stable")
        support, first = np.unique(mean[order], return_index=True)
        probabilities = quantiles[i, order[first]]
        sigma = np.sqrt(variance)
        lo = np.interp(mean - sigma, support, probabilities, left=1 / (n + 1), right=n / (n + 1))
        hi = np.interp(mean + sigma, support, probabilities, left=1 / (n + 1), right=n / (n + 1))
        mapped_variances[i] = (0.5 * (ndtri(hi) - ndtri(lo))) ** 2
    return transformed, mapped_variances


def scale_target(target, weight):
    mean = np.average(target, weights=weight)
    scale = np.sqrt(np.average((target - mean) ** 2, weights=weight))
    scale = scale if scale > 1e-8 else 1.0
    return (target - mean) / scale, mean, scale


def score(target, prediction):
    residual = float(np.mean((target - prediction) ** 2))
    target_var, prediction_var = np.var(target), np.var(prediction)
    variable = len(target) > 1 and min(target_var, prediction_var) > 1e-12
    return {"mse": residual,
            "r2": 1 - residual / target_var if target_var > 1e-12 else None,
            "pearson": float(np.corrcoef(target, prediction)[0, 1]) if variable else None,
            "spearman": float(spearmanr(target, prediction).statistic) if variable else None}


def fit_student(dosage, targets, variances, train_indices, test_indices, *, mode="probabilistic",
                additive=False, bootstraps=5, alpha=None, l1_ratio=0.5, cv=3, alphas=15,
                max_iter=10000, sigma_floor=1e-4, seed=42):
    """Fit transformed targets on the fixed 80% fold; never refit on test donors.

    Targets are member-by-donor in probabilistic mode and one-by-donor in
    deterministic mode. Every supplied SNP enters the elastic-net design;
    there is no feature ranking or cap. Returned coefficients use raw BIM A1
    dosage units and retain the fitted nonzero support. HWE projection uses
    allele frequencies from all reference genotypes, shared across draws.
    """
    dosage, targets = np.asarray(dosage, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    train_indices, test_indices = np.asarray(train_indices), np.asarray(test_indices)
    if (dosage.ndim != 2 or targets.ndim != 2 or targets.shape[1] != len(dosage)
            or not np.isin(dosage, [0, 1, 2]).all() or not np.isfinite(targets).all()):
        raise ValueError("Expected complete 0/1/2 dosages and aligned finite teacher targets")
    if mode not in ("probabilistic", "deterministic") or (mode == "deterministic" and len(targets) != 1):
        raise ValueError("Deterministic mode requires one ensemble-mean target")
    if len(train_indices) < 2 or not len(test_indices) or set(train_indices) & set(test_indices):
        raise ValueError("Expected disjoint, nonempty training and test folds")
    n_snps = dosage.shape[1]
    if not n_snps:
        raise ValueError("No SNPs available")
    if mode == "probabilistic":
        if variances is None or np.shape(variances) != targets.shape:
            raise ValueError("Probabilistic mode requires one aleatoric variance per target")
        variances = np.asarray(variances, dtype=np.float64)
        if not np.isfinite(variances).all() or (variances < 0).any():
            raise ValueError("Invalid teacher variances")
        precision = 1 / np.maximum(variances[:, train_indices], sigma_floor ** 2)
    else:
        precision = np.ones((1, len(train_indices)))
        bootstraps = 1
    design = dosage if additive else np.concatenate([dosage, dosage == 1], axis=1)
    y = targets[:, train_indices]
    x = design[train_indices]
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    if not np.any(x_scale > 1e-8):
        raise ValueError("No variable genotype features in the training fold")
    # Unit scale retains constant columns without dividing by zero.
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
    x_scaled = np.asfortranarray((x - x_mean) / x_scale)
    if alpha is None:
        reference = y.mean(axis=0)
        reference_precision = 1 / np.mean(1 / precision, axis=0)
        standardized, _, _ = scale_target(reference, reference_precision)
        if np.var(standardized) < 1e-12:
            alpha = 1.0
        else:
            search = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=alphas, cv=min(cv, len(x)),
                                  max_iter=max_iter, n_jobs=1, precompute=False, selection="cyclic")
            search.fit(x_scaled, standardized, sample_weight=reference_precision)
            alpha = float(search.alpha_)
    n_members, n_train = y.shape
    rng = np.random.default_rng(seed)
    counts = (rng.multinomial(n_train, np.full(n_train, 1 / n_train), size=(n_members, bootstraps))
              if mode == "probabilistic" else np.ones((1, 1, n_train), dtype=int))
    coefficients = np.empty((n_members, bootstraps, design.shape[1]))
    intercepts = np.empty((n_members, bootstraps))
    iterations = np.empty((n_members, bootstraps), dtype=int)
    for member in range(n_members):
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                               selection="cyclic", warm_start=True, precompute=False)
        for bootstrap in range(bootstraps):
            weight = counts[member, bootstrap] * precision[member]
            standardized, y_mean, y_scale = scale_target(y[member], weight)
            if np.var(standardized[weight > 0]) < 1e-12:
                coefficients[member, bootstrap] = 0
                intercepts[member, bootstrap] = y_mean
                iterations[member, bootstrap] = 0
                continue
            estimator.fit(x_scaled, standardized, sample_weight=weight)
            coefficients[member, bootstrap] = y_scale * estimator.coef_ / x_scale
            intercepts[member, bootstrap] = y_mean + y_scale * estimator.intercept_ - x_mean @ coefficients[member, bootstrap]
            iterations[member, bootstrap] = estimator.n_iter_
    beta = coefficients[:, :, :n_snps]
    delta = np.zeros_like(beta) if additive else coefficients[:, :, n_snps:]
    mean_g = np.einsum("mbn,np->mbp", counts, dosage[train_indices]) / n_train
    mean_h = np.einsum("mbn,np->mbp", counts, dosage[train_indices] == 1) / n_train
    frequencies = np.broadcast_to(dosage.mean(axis=0) / 2, beta.shape)
    projected = beta + (1 - 2 * frequencies) * delta
    projected_intercepts = intercepts + np.sum(mean_g * (beta - projected) + mean_h * delta, axis=2)
    # Keep full G+H support, including heterozygote effects whose additive projection cancels.
    keep = np.any((np.abs(beta) > 1e-12) | (np.abs(delta) > 1e-12), axis=(0, 1))
    full_predictions = (intercepts[:, :, None] + np.einsum("np,mbp->mbn", dosage, beta)
                        + np.einsum("np,mbp->mbn", dosage == 1, delta))
    projected_predictions = projected_intercepts[:, :, None] + np.einsum("np,mbp->mbn", dosage, projected)
    evaluation = {}
    for name, indices in (("train", train_indices), ("test", test_indices)):
        target = targets[:, indices].mean(axis=0)
        for predictor, predictions in (("full", full_predictions), ("additive_projection", projected_predictions)):
            prediction = predictions[:, :, indices].mean(axis=(0, 1))
            evaluation[f"{name}_{predictor}"] = score(target, prediction)
    coefficient_draws = projected[:, :, keep]
    return {
        "snp_indices": np.flatnonzero(keep), "coefficient_draws": coefficient_draws,
        "intercept_draws": projected_intercepts, "dosage_draws": beta[:, :, keep],
        "heterozygote_draws": delta[:, :, keep], "full_intercept_draws": intercepts,
        "allele_frequency_draws": frequencies[:, :, keep], "bootstrap_counts": counts,
        "coefficients": coefficient_draws.mean(axis=(0, 1)),
        "intercept": float(projected_intercepts.mean()),
        "coefficient_variances": coefficient_draws.var(axis=(0, 1)),
        "inclusion_probabilities": (np.abs(coefficient_draws) > 1e-12).mean(axis=(0, 1)),
        "alpha": float(alpha), "iterations": iterations, "evaluation": evaluation,
    }
