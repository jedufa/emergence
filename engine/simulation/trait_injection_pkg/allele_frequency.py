from __future__ import annotations

"""simulation/trait_injection_pkg/allele_frequency.py

Phase 5B: allele frequency tracking.

Requirement
-----------
- Multi-allelic support.
- Sparse thresholding.

This is designed to be deterministic and safe for logging/metrics.
It does not attempt to infer alleles from continuous values beyond stable
stringification.

No RNG. No wall clock.
"""

from typing import Any, Dict, Iterable, List
import json


SPARSE_BUCKET_KEY = "__sparse__"


def _as_json_key(value: Any) -> str:
    """Convert an allele value into a stable string key.

    Notes
    -----
    - Floats are rounded to a fixed precision to reduce tiny numeric noise.
    - Lists/dicts are serialized as canonical JSON.
    """
    try:
        # numpy scalar compatibility
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, float):
        # Deterministic rounding (do not depend on locale).
        v = round(float(value), 8)
        s = ("%.8f" % v).rstrip("0").rstrip(".")
        return s if s else "0"

    if isinstance(value, (int, bool, str)):
        return str(value)

    # Fallback: canonical JSON if possible, else repr.
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return repr(value)


def compute_allele_frequencies(
    genomes: Iterable[Any],
    *,
    locus_tags: List[str],
    sparse_threshold: float = 0.0,
    sparse_bucket_key: str = SPARSE_BUCKET_KEY,
) -> Dict[str, Dict[str, float]]:
    """Compute allele frequencies for a list of genomes.

    Parameters
    ----------
    genomes:
        Iterable of genome-like objects.
    locus_tags:
        Genome attribute names to count.
    sparse_threshold:
        Minimum frequency to keep an allele as its own key.
        Alleles below this are aggregated into sparse_bucket_key.
    sparse_bucket_key:
        Aggregation key.

    Returns
    -------
    dict
        locus_tag -> { allele_key -> frequency }

    Determinism
    -----------
    - Iteration order of genomes does not matter (counts only).
    - Allele keys are stable via _as_json_key.
    """
    tags = [str(t) for t in (locus_tags or []) if str(t)]
    if not tags:
        return {}

    counts: Dict[str, Dict[str, int]] = {t: {} for t in tags}
    total = 0
    for g in genomes:
        if g is None:
            continue
        total += 1
        for t in tags:
            try:
                v = getattr(g, t)
            except Exception:
                continue
            k = _as_json_key(v)
            d = counts[t]
            d[k] = int(d.get(k, 0)) + 1

    if total <= 0:
        return {t: {} for t in tags}

    thr = float(sparse_threshold)
    if thr < 0.0:
        thr = 0.0
    if thr > 1.0:
        thr = 1.0

    out: Dict[str, Dict[str, float]] = {}
    for t in tags:
        freqs: Dict[str, float] = {}
        sparse_sum = 0.0
        for k in sorted(counts[t].keys()):
            f = float(counts[t][k]) / float(total)
            if f < thr:
                sparse_sum += f
            else:
                freqs[k] = f
        if sparse_sum > 0.0:
            freqs[str(sparse_bucket_key)] = sparse_sum
        out[t] = freqs

    return out
