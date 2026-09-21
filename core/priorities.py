"""The four institutional priorities and the short list builder.

P1 Yield, P2 expected net tuition revenue and P4 academic strength are
individual scores. P3 Diversity is set-dependent (SR-5): no individual is
diverse, so it is scored as marginal movement toward composition targets and
requires greedy selection.

Each of P1, P2 and P4 is converted to percentile rank before blending, because a
probability, a dollar amount and a similarity are not otherwise commensurable.
Percentile rank is monotone, so any single priority induces the same ordering it
would have raw, which is what AC-6 depends on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg               # noqa: E402
from core.loader import Dataset    # noqa: E402

SUCCESS_COMPONENTS = {
    "Attainment": "success_attainment",
    "Growth": "success_growth",
    "Completion": "success_completion",
    "Contribution": "success_contribution",
}


def percentile(x: np.ndarray) -> np.ndarray:
    """Rank-based percentile in [0,1]. Monotone, so it preserves ordering."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average ranks within ties so equal inputs get equal percentiles
    s = pd.Series(x)
    ranks = s.rank(method="average").to_numpy() - 1.0
    return (ranks / (n - 1)).astype(np.float32)


# ------------------------------------------------------------------ P1, P2
def p1_yield(ds: Dataset) -> np.ndarray:
    return np.asarray(ds.applicants.predicted_yield_prob, dtype=np.float32)


def p2_expected_ntr(ds: Dataset) -> np.ndarray:
    """SR-6: expected, not nominal. A full-pay applicant who will not enrol
    contributes nothing. P1 and P2 therefore share a factor and are not
    independent, which the interface must state at the controls."""
    net = np.asarray(ds.applicants.net_price, dtype=np.float32)
    prob = np.asarray(ds.applicants.predicted_yield_prob, dtype=np.float32)
    return net * prob


# ---------------------------------------------------------------------- P4
def success_scores(ds: Dataset, enabled: dict[str, bool] | None = None
                   ) -> np.ndarray:
    """Weighted mean of the toggled DR-4 success components (requirements 1.4).

    Never collapsed into one stored number: a single blended score conceals
    which definition of success is doing the work, and the four definitions
    select different people.
    """
    # An explicit dict is authoritative: a missing key means OFF. Defaulting
    # absent keys to True would silently re-enable components the user turned
    # off, which is the IR-1b failure mode in a different costume.
    if enabled is None:
        enabled = {k: True for k in SUCCESS_COMPONENTS}
    cols = [c for label, c in SUCCESS_COMPONENTS.items() if enabled.get(label, False)]
    if not cols:
        return np.zeros(ds.n_alum, dtype=np.float32)
    stack = np.stack([np.asarray(ds.alumni[c], dtype=np.float32) for c in cols])
    return stack.mean(axis=0)


def p4_academic(sim: np.ndarray, ds: Dataset,
                success_enabled: dict[str, bool] | None = None,
                use_credential: bool = True,
                use_validated: bool = True,
                k: int | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Requirements 1.3. Returns (score_percentile, thin_mask, neighbour_idx, parts).

    `parts` carries the raw pre-percentile components so the thin-neighbourhood
    guard is inspectable. The returned percentile is normalised pool-wide per
    design 5.3, which necessarily couples every applicant's rank to every other;
    the raw components are where "validation was withheld" is observable, and
    where XR-2 reads the disclosure from.

    P4a credential  - reader rating and recalculated GPA. The conventional
                      measure, kept visible for comparison.
    P4b validated   - similarity-weighted kNN regression on the alumni success
                      label. Chosen because it is nameable: the answer to "why
                      does this applicant score well" is a list of alumni, which
                      SR-1a demands and an embedding cannot give.

    Thin-neighbourhood guard (design 5.5): if every one of an applicant's top-k
    alumni falls below the similarity floor, P4b rests on no real evidence. It is
    withheld, P4 falls back to P4a, and the caller must disclose that. This is
    the common case for a rare major with no alumni analogue, and a confident
    score from thin evidence would be the most dangerous failure here.
    """
    k = k or cfg.TOPK_P4
    rating = np.asarray(ds.applicants.academic_rating, dtype=np.float32)
    gpa = np.asarray(ds.applicants.hs_gpa_recalc, dtype=np.float32)
    cred = 0.5 * percentile(rating) + 0.5 * percentile(gpa)

    k = int(min(k, sim.shape[1]))
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-s, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    s = np.take_along_axis(s, order, axis=1)

    success = success_scores(ds, success_enabled)
    num = (s * success[idx]).sum(axis=1)
    den = s.sum(axis=1)
    validated = np.where(den > cfg.EPS, num / np.maximum(den, cfg.EPS), 0.0)
    thin = s.max(axis=1) < cfg.SIM_FLOOR

    val_pct = percentile(validated)
    if use_credential and use_validated:
        score = np.where(thin, cred, 0.5 * cred + 0.5 * val_pct)
    elif use_validated:
        score = np.where(thin, cred, val_pct)
    else:
        score = cred
    parts = {"credential": cred.astype(np.float32),
             "validated": validated.astype(np.float32),
             "validated_pct": val_pct.astype(np.float32),
             "raw": score.astype(np.float32),
             "neighbour_sim": s.astype(np.float32)}
    return percentile(score.astype(np.float32)), thin, idx, parts


# ---------------------------------------------------------------------- P3
def normalise_targets(targets: dict[str, dict[str, float]]
                      ) -> dict[str, dict[str, float]]:
    """Each dimension's shares must sum to 1 for the L1 distance in
    `build_shortlist` to be the sum of per-dimension distances. User-edited
    targets are renormalised rather than rejected."""
    out: dict[str, dict[str, float]] = {}
    for field, spec in targets.items():
        total = sum(max(0.0, v) for v in spec.values())
        if total <= 0:
            n = len(spec) or 1
            out[field] = {k: 1.0 / n for k in spec}
        else:
            out[field] = {k: max(0.0, v) / total for k, v in spec.items()}
    return out


def composition_matrix(ds: Dataset,
                       targets: dict[str, dict[str, float]] | None = None
                       ) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """One-hot C over every target category, plus the target vector.

    Each configured dimension contributes categories whose target proportions
    sum to 1, so an L1 distance over the whole vector is the sum of per-dimension
    L1 distances. Counting represented regions is useless as a diversity metric
    (requirements 2.4); distance from a target composition is not.
    """
    spec_all = normalise_targets(targets or cfg.COMPOSITION_TARGETS)
    labels: list[tuple[str, str]] = []
    target_vec: list[float] = []
    columns: list[np.ndarray] = []

    for field, spec in spec_all.items():
        values = np.asarray(ds.applicants[field].astype(str))
        for category, share in spec.items():
            labels.append((field, category))
            target_vec.append(float(share))
            columns.append((values == category).astype(np.float32))

    C = np.stack(columns, axis=1)
    return C, np.asarray(target_vec, dtype=np.float32), labels


def composition_of(C: np.ndarray, picked: np.ndarray,
                   labels: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Realised proportions per dimension for a selected set."""
    if picked.sum() == 0:
        return {}
    counts = C[picked].sum(axis=0)
    n = int(picked.sum())
    out: dict[str, dict[str, float]] = {}
    for (field, category), c in zip(labels, counts):
        out.setdefault(field, {})[category] = float(c) / n
    return out


# ---------------------------------------------------------- the short list
def _stable_argmax(score: np.ndarray, id_rank: np.ndarray) -> int:
    """NFR-5: np.argmax breaks ties by array position, which makes output depend
    on CSV row order. Ties break on applicant_id instead."""
    best = score.max()
    cands = np.flatnonzero(score >= best - 1e-9)
    return int(cands[np.argmin(id_rank[cands])])


def stable_top_n(score: np.ndarray, n: int, id_rank: np.ndarray) -> np.ndarray:
    order = np.lexsort((id_rank, -score))
    picked = np.zeros(score.size, dtype=bool)
    picked[order[:n]] = True
    return picked


def build_shortlist(indiv_pct: np.ndarray,
                    C: np.ndarray,
                    target: np.ndarray,
                    B: np.ndarray,
                    at_risk: np.ndarray | None,
                    w_p3: float,
                    w_club: float,
                    size: int,
                    id_rank: np.ndarray) -> np.ndarray:
    """SR-5b and SR-8c. Returns a boolean mask over applicants.

    Short-circuits to a plain sort when Diversity carries no weight. That is
    both the NFR-1 optimisation and the reason AC-6 reproduces the other three
    priorities exactly rather than approximately.

    Greedy is not globally optimal; each pick is locally best given prior picks.
    That is inherent, conceded by AC-6, and must be stated in the interface.
    """
    n = indiv_pct.size
    size = int(min(size, n))
    if w_p3 <= 0.0:
        return stable_top_n(indiv_pct, size, id_rank)

    if at_risk is None:
        at_risk = np.ones(B.shape[1], dtype=np.float32)

    chosen = np.zeros(n, dtype=bool)
    counts = np.zeros(C.shape[1], dtype=np.float32)
    covered = np.zeros(B.shape[1], dtype=np.float32)

    for k in range(size):
        # vectorised over all candidates; the loop is over picks, never candidates
        d_new = np.abs((counts + C) / (k + 1) - target).sum(axis=1)
        gain_comp = -d_new
        gain_club = (B * at_risk * (1.0 - covered)).sum(axis=1)
        raw = gain_comp + w_club * gain_club

        free = ~chosen
        lo, hi = raw[free].min(), raw[free].max()
        gain_p3 = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

        score = indiv_pct + w_p3 * gain_p3
        score = np.where(free, score, -np.inf)
        pick = _stable_argmax(score, id_rank)

        chosen[pick] = True
        counts += C[pick]
        if B.shape[1]:
            covered = np.maximum(covered, B[pick])
    return chosen


def blend_individual(p1p: np.ndarray, p2p: np.ndarray, p4p: np.ndarray,
                     w_yield: float, w_ntr: float, w_acad: float) -> np.ndarray:
    """Weighted mean of the three individual priorities, normalised so the blend
    stays on [0,1] as weights change (same reasoning as SR-1b)."""
    total = w_yield + w_ntr + w_acad
    if total <= 0.0:
        return np.zeros_like(p1p)
    return ((w_yield * p1p + w_ntr * p2p + w_acad * p4p) / total).astype(np.float32)


if __name__ == "__main__":
    import time

    from core.clubs import coverage_matrix, project_rosters
    from core.features import build_tensors, flatten
    from core.loader import load
    from core.similarity import contract, weights_from_words

    ds = load()
    M, A = build_tensors(ds)
    M2, A2 = flatten(M, A)
    sim = contract(M2, A2, weights_from_words(cfg.DIM_DEFAULTS),
                   (ds.n_app, ds.n_alum))

    p1 = percentile(p1_yield(ds))
    p2 = percentile(p2_expected_ntr(ds))
    p4, thin, nbrs, parts = p4_academic(sim, ds)
    print(f"P1 mean {p1.mean():.3f}  P2 mean {p2.mean():.3f}  P4 mean {p4.mean():.3f}")
    print(f"thin neighbourhoods: {int(thin.sum())} of {ds.n_app} "
          f"({thin.mean():.1%}) fall back to credentials")

    C, target, labels = composition_matrix(ds)
    ros = project_rosters(ds)
    B, club_ids = coverage_matrix(ds, ros)
    id_rank = np.argsort(np.argsort(np.asarray(ds.app_ids)))
    print(f"C {C.shape}  target sums {target.sum():.1f} over "
          f"{len(cfg.COMPOSITION_TARGETS)} dimensions  B {B.shape}")

    t = time.perf_counter()
    indiv = blend_individual(p1, p2, p4, 0.5, 0.5, 0.5)
    mask = build_shortlist(indiv, C, target, B, None, 0.5, 0.25,
                           cfg.CLASS_TARGET, id_rank)
    greedy_ms = (time.perf_counter() - t) * 1000
    print(f"\ngreedy shortlist {int(mask.sum())} picks in {greedy_ms:.0f} ms")

    t = time.perf_counter()
    mask0 = build_shortlist(indiv, C, target, B, None, 0.0, 0.25,
                            cfg.CLASS_TARGET, id_rank)
    print(f"short-circuit (P3 off) {int(mask0.sum())} picks in "
          f"{(time.perf_counter() - t) * 1000:.1f} ms")
    print(f"greedy vs sorted differ: {int((mask != mask0).sum())} applicants "
          f"(AC-14 needs > 0)")

    print("\ncoverage of at-risk clubs")
    for w3 in (0.0, 0.5, 2.5):
        m = build_shortlist(indiv, C, target, B, None, w3, 1.0,
                            cfg.CLASS_TARGET, id_rank)
        cov = int((B[m].sum(axis=0) > 0).sum())
        print(f"  P3 weight {w3:4.2f}: {cov} of {B.shape[1]} at-risk clubs covered")
