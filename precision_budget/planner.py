from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    name: str
    excluded: tuple[str, ...]
    accuracy: float
    latency_ms: float
    size_bytes: int


def choose(baseline: Result, candidates: list[Result], max_accuracy_drop: float, min_speedup: float = 0.05) -> Result:
    if not 0 <= max_accuracy_drop <= 1:
        raise ValueError("max_accuracy_drop must be between 0 and 1")
    if not 0 <= min_speedup < 1:
        raise ValueError("min_speedup must be between 0 and 1")
    eligible = [c for c in candidates if baseline.accuracy - c.accuracy <= max_accuracy_drop
                and c.latency_ms <= baseline.latency_ms * (1 - min_speedup)]
    # Avoid selecting a variant based on microsecond-scale timing noise.
    return min(eligible, key=lambda c: (c.latency_ms, c.size_bytes)) if eligible else baseline


def sensitivity_order(full: Result, probes: list[Result]) -> list[str]:
    """Rank nodes by accuracy recovered when that node stays in FP32."""
    return [p.excluded[0] for p in sorted(probes, key=lambda p: (p.accuracy - full.accuracy, -p.latency_ms), reverse=True)]
