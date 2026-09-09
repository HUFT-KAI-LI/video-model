"""Small target-level statistics shared by the diagnostics (no model dependency).

Every statistic treats the unique target as the independent unit: noise seeds are
repeated measures inside a target, so bootstrap resampling is over targets.
"""
import random
import statistics


def bootstrap_ci(values, samples=10000, seed=0):
    """Percentile bootstrap over per-target values."""
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0, "positive": 0}
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    low = means[max(0, int(.025 * samples))]
    high = means[min(samples - 1, int(.975 * samples))]
    return {"mean": statistics.mean(values), "low": low, "high": high, "n": count,
            "positive": sum(value > 0 for value in values)}


def auroc(scores, labels):
    """AUROC via pairwise wins (labels: 1 = positive, 0 = negative); ties count 0.5."""
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    wins = 0.0
    for first in positive:
        for second in negative:
            wins += 1.0 if first > second else (0.5 if first == second else 0.0)
    return wins / (len(positive) * len(negative))


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def fraction_true(values):
    values = list(values)
    return (sum(bool(value) for value in values) / len(values)) if values else None
