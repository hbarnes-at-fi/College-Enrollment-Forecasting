"""Weighted-mean similarity and its exact decomposition (SR-1).

The whole of SR-1 is one expression:

    sim(a,e) = sum_d w_d * M[a,e,d]  /  max(sum_d w_d * A[a,e,d], eps)

One formula, four requirements:

    SR-1   additive weighted mean          numerator is linear in w
    SR-1b  comparable as weights change    division by the masked weight sum
    IR-1b  `Not considered` truly removes   w_d = 0 cancels from both terms
    SR-7b  inapplicable dimensions dropped  A[a,e,d] = 0 removes one pair's term
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402


def weights_from_words(words: dict[str, str]) -> np.ndarray:
    """Map the nine word-scale dimension settings to a weight vector."""
    return np.array([cfg.WEIGHT[words.get(name, "Moderate")]
                     for name in cfg.DIM_NAMES], dtype=np.float32)


def contract(M2: np.ndarray, A2: np.ndarray, w: np.ndarray,
             shape: tuple[int, int]) -> np.ndarray:
    """Re-weight the precomputed tensor. This is the whole interaction hot path."""
    num = M2 @ w
    den = A2 @ w
    out = np.where(den > cfg.EPS, num / np.maximum(den, cfg.EPS), 0.0)
    return out.reshape(shape).astype(np.float32)


def decompose(M: np.ndarray, A: np.ndarray, w: np.ndarray,
              a_idx: int, e_idx: int) -> dict:
    """XR-1. Contributions sum exactly to the score, by construction.

    Returns contributions, the raw per-dimension match, and which dimensions
    were dropped for this pair under SR-7b, which XR-2 must disclose.
    """
    m = M[a_idx, e_idx, :].astype(np.float64)
    a = A[a_idx, e_idx, :].astype(np.float64)
    den = float((a * w).sum())

    contribs: dict[str, float] = {}
    matches: dict[str, float] = {}
    dropped: list[str] = []
    for d, name in enumerate(cfg.DIM_NAMES):
        if w[d] <= 0.0:
            continue                      # IR-1b: not considered, not reported
        if a[d] <= 0.0:
            dropped.append(name)          # SR-7b: nothing to compare
            continue
        contribs[name] = (w[d] * m[d] / den) if den > cfg.EPS else 0.0
        matches[name] = m[d]

    score = sum(contribs.values())
    return {"score": score, "contributions": contribs,
            "matches": matches, "dropped": dropped}


def topk(sim: np.ndarray, k: int, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """NR-3a and NR-3b: at most k per applicant, and never below the floor.

    Returns (indices, scores), both (n_app, k), with -1 / nan where an applicant
    has fewer than k matches clearing the floor. A weak top match is not a match.
    """
    n_app, n_alum = sim.shape
    k = int(min(k, n_alum))
    part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(sim, part, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(part, order, axis=1)
    val = np.take_along_axis(vals, order, axis=1)

    keep = val >= floor
    idx = np.where(keep, idx, -1)
    val = np.where(keep, val, np.nan)
    return idx, val


def percentile_of(sim_row: np.ndarray, value: float) -> float:
    """Where a score sits within its own row, for the XR-2 strength band."""
    finite = sim_row[np.isfinite(sim_row)]
    if finite.size == 0:
        return 0.0
    return float((finite <= value).mean())


if __name__ == "__main__":
    import time

    from core.features import build_tensors, flatten
    from core.loader import load

    ds = load()
    M, A = build_tensors(ds)
    M2, A2 = flatten(M, A)

    w = weights_from_words(cfg.DIM_DEFAULTS)
    print(f"default weights {dict(zip(cfg.DIM_NAMES, w))}")

    t = time.perf_counter()
    for _ in range(20):
        sim = contract(M2, A2, w, (ds.n_app, ds.n_alum))
    print(f"contract        {(time.perf_counter() - t) / 20 * 1000:6.1f} ms"
          f"   (NFR-1 budget 300 ms total)")

    t = time.perf_counter()
    for _ in range(20):
        idx, val = topk(sim, cfg.TOPK_EDGES, cfg.SIM_FLOOR)
    print(f"topk            {(time.perf_counter() - t) / 20 * 1000:6.1f} ms")

    print(f"sim mean {sim.mean():.3f}  max {sim.max():.3f}")
    print(f"edges kept at floor {cfg.SIM_FLOOR}: {int((idx >= 0).sum())} "
          f"of {ds.n_app * cfg.TOPK_EDGES} possible")

    # XR-1: contributions must sum to the score exactly
    a_i = 0
    e_i = int(idx[a_i, 0])
    dec = decompose(M, A, w, a_i, e_i)
    print(f"\ndecomposition for {ds.app_ids[a_i]} vs {ds.alum_ids[e_i]}")
    for name, c in sorted(dec["contributions"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} contrib {c:.4f}  match {dec['matches'][name]:.3f}")
    print(f"  dropped (SR-7b): {dec['dropped'] or 'none'}")
    print(f"  sum of contributions {dec['score']:.9f}")
    print(f"  contract() score     {sim[a_i, e_i]:.9f}")
    print(f"  identical: {abs(dec['score'] - sim[a_i, e_i]) < 1e-6}")

    # IR-1b: a zero weight must be genuinely inert
    w0 = w.copy()
    w0[2] = 0.0
    s0 = contract(M2, A2, w0, (ds.n_app, ds.n_alum))
    print(f"\nzeroing 'Activity portfolio' changes {int((s0 != sim).sum())} "
          f"of {sim.size} scores (expected: most)")
    dec0 = decompose(M, A, w0, a_i, e_i)
    print(f"  dimension absent from decomposition: "
          f"{'Activity portfolio' not in dec0['contributions']}")
