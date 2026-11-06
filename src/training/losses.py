"""Gene-wise Pearson/centered cosine and donor-level Poisson/Gaussian NLL."""

import torch


def correlation_terms(mean, target, observed):
    """Center across genes (axis 0) separately for every observed donor/cell."""
    count = observed.sum(dim=0, keepdim=True).clamp_min(1)
    p = torch.where(observed, mean, 0)
    t = torch.where(observed, target, 0)
    p = torch.where(observed, p - p.sum(dim=0, keepdim=True) / count, 0)
    t = torch.where(observed, t - t.sum(dim=0, keepdim=True) / count, 0)
    numerator = (p * t).sum(dim=0)
    denominator = p.norm(dim=0) * t.norm(dim=0)
    valid = (count.squeeze(0) > 1) & (denominator > 1e-8)
    weight = valid.to(mean.dtype)
    n_valid = weight.sum().clamp_min(1)
    pc = -((numerator / denominator.clamp_min(1e-8)) * weight).sum() / n_valid
    cosine = ((1 - numerator / (denominator + 1e-8)) * weight).sum() / n_valid
    return pc, cosine


def loss_terms(mean, variance, target, observed):
    """Targets/masks are gene × donor × cell; predictions are gene × cell."""
    mean = mean[:, None, :].expand_as(target)
    variance = variance[:, None, :].expand_as(target)
    target = torch.where(observed, target, 0)
    pc, cosine = correlation_terms(mean, target, observed)
    count = observed.sum().clamp_min(1)
    variance = variance.clamp_min(1e-8)
    pnll = ((mean - target * mean.clamp_min(1e-8).log()) * observed).sum() / count
    gnll = (0.5 * (variance.log() + (target - mean).square() / variance) * observed).sum() / count
    return {"pc": pc, "cosine": cosine, "pnll": pnll, "gnll": gnll}


def ensemble_loss(model, features, target, observed, weights):
    means, variances = model.member_predictions(features)
    terms = [loss_terms(mean, variance, target, observed) for mean, variance in zip(means, variances)]
    # Sum member objectives: members have independent gradients and parameters.
    loss = sum(weights[name] * value for member in terms for name, value in member.items())
    metrics = {name: torch.stack([member[name].detach() for member in terms]).mean().item() for name in weights}
    return loss, metrics
