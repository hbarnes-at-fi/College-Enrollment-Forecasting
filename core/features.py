"""Build the per-dimension match tensor and applicability mask (specs §6).

    M: (n_app, n_alum, 9) float32   match score in [0,1]
    A: (n_app, n_alum, 9) float32   1.0 where the dimension is meaningful

`A` serves SR-7b. Where a dimension cannot be compared for a pair, it leaves
that pair's normaliser entirely rather than being scored zero (a false penalty)
or one (a false inflation).

Everything is vectorised; a Python loop over 720,000 pairs x 9 dimensions is not
viable inside the NFR-1 budget even as a one-time build.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg               # noqa: E402
from core.loader import Dataset    # noqa: E402

# d=4, leadership. Rank of the most senior role held.
ROLE_RANK = {"Member": 0.0, "Officer": 2.0, "Captain": 2.0,
             "President": 3.0, "Founder": 3.0}
CLUB_ROLE_RANK = {"Member": 0.0, "Officer": 2.0, "President": 3.0, "Founder": 3.0}

INCOME_ORDER = ["Under $30K", "$30K - $59K", "$60K - $99K", "$100K - $149K",
                "$150K - $249K", "$250K and above", "Not Reported"]
GPA_BANDS = [2.75, 3.00, 3.25, 3.50, 3.75, 4.00]   # 7 bands with the tail


def dimension_names() -> list[str]:
    return list(cfg.DIM_NAMES)


# ----------------------------------------------------------------- helpers
def _codes(values, vocabulary: list[str]) -> np.ndarray:
    lookup = {v: i for i, v in enumerate(vocabulary)}
    return np.array([lookup.get(v, -1) for v in values], dtype=np.int16)


def _band(values, edges: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return np.searchsorted(np.asarray(edges), v).astype(np.int16)


def _match_ordinal(a: np.ndarray, b: np.ndarray, span: float) -> np.ndarray:
    """1.0 at equality, falling linearly to 0.0 at `span` apart."""
    d = np.abs(a[:, None].astype(np.float32) - b[None, :].astype(np.float32))
    return np.clip(1.0 - d / span, 0.0, 1.0).astype(np.float32)


def _match_exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, None] == b[None, :]).astype(np.float32)


def _weighted_sets(ids: list[str], universe: list[str],
                   sets: dict[str, set[str]],
                   weights: dict[str, dict[str, float]],
                   idf: dict[str, float],
                   drop: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (sqrt-scaled matrix for the intersection matmul, per-person norms).

    Uses the extended (Tanimoto) Jaccard so the intersection is a single matrix
    product:

        inter = S_a . S_e            with S[p,x] = sqrt(idf[x] * rho_p[x])
        J     = inter / (norm_a + norm_e - inter)

    This reduces to classic Jaccard when all role weights are 1, and keeps the
    build inside a matmul instead of a (n_app, n_alum, n_items) broadcast that
    would need hundreds of megabytes.
    """
    drop = drop or set()
    kept = [x for x in universe if x not in drop]
    col = {x: i for i, x in enumerate(kept)}
    S = np.zeros((len(ids), len(kept)), dtype=np.float32)
    norms = np.zeros(len(ids), dtype=np.float32)
    for i, pid in enumerate(ids):
        for item in sets.get(pid, ()):  # noqa: SIM118
            j = col.get(item)
            if j is None:
                continue
            rho = weights.get(pid, {}).get(item, 1.0)
            w = idf.get(item, 0.0) * rho
            S[i, j] = np.sqrt(max(0.0, w))
            norms[i] += w
    return S, norms


def _tanimoto(Sa, na, Se, ne) -> tuple[np.ndarray, np.ndarray]:
    inter = (Sa @ Se.T).astype(np.float32)
    union = na[:, None] + ne[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.where(union > cfg.EPS, inter / np.maximum(union, cfg.EPS), 0.0)
    # applicable only where at least one side has something to compare (SR-7b)
    applicable = ((na[:, None] > 0) | (ne[None, :] > 0)).astype(np.float32)
    return np.clip(j, 0.0, 1.0).astype(np.float32), applicable


def _leadership(ids: list[str], ds: Dataset) -> np.ndarray:
    """Most senior role held plus a credit for breadth of leadership."""
    score = np.zeros(len(ids), dtype=np.float32)
    idx = {p: i for i, p in enumerate(ids)}
    top = {p: 0.0 for p in ids}
    count = {p: 0 for p in ids}
    for r in ds.act_edges.itertuples(index=False):
        if r.person_id in top:
            rank = ROLE_RANK.get(r.role, 0.0)
            top[r.person_id] = max(top[r.person_id], rank)
            count[r.person_id] += rank > 0
    for r in ds.club_edges.itertuples(index=False):
        if r.person_id in top and r.relationship == "MEMBER_OF":
            rank = CLUB_ROLE_RANK.get(str(r.club_role), 0.0)
            top[r.person_id] = max(top[r.person_id], rank)
            count[r.person_id] += rank > 0
    for p, i in idx.items():
        score[i] = top[p] + 0.5 * min(count[p], 4)
    return score


def _athletics(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """specs §4.4 tiers. Mutual non-participation is NOT a match (SR-7b)."""
    app_sport = np.asarray(ds.applicants.athletic_recruit_sport.fillna("").astype(str))
    alum_sport = np.asarray(ds.alumni.entry_athletic_recruit_sport.fillna("").astype(str))
    app_cap = np.asarray(ds.applicants.coach_support_tier.astype(str)) == "Slot"
    alum_cap = np.asarray(ds.alumni.team_captain.astype(str)) == "Y"

    a_has = (app_sport != "") & (app_sport != "nan")
    e_has = (alum_sport != "") & (alum_sport != "nan")
    same = (app_sport[:, None] == alum_sport[None, :]) & a_has[:, None] & e_has[None, :]
    both = a_has[:, None] & e_has[None, :]
    one = a_has[:, None] ^ e_has[None, :]
    both_cap = app_cap[:, None] & alum_cap[None, :]

    m = np.zeros((len(app_sport), len(alum_sport)), dtype=np.float32)
    m[both & ~same] = 0.60
    m[same] = 0.85
    m[same & both_cap] = 1.00
    m[one] = 0.20
    applicable = (both | one).astype(np.float32)
    return m, applicable


# ------------------------------------------------------------------- build
def build_tensors(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    n_app, n_alum = ds.n_app, ds.n_alum
    M = np.zeros((n_app, n_alum, cfg.N_DIM), dtype=np.float32)
    A = np.ones((n_app, n_alum, cfg.N_DIM), dtype=np.float32)

    app, alum = ds.applicants, ds.alumni

    # ---- d0 academic preparation: reader rating and banded GPA
    r_app = np.asarray(app.academic_rating, dtype=np.int16)
    r_alum = np.asarray(alum.entry_academic_rating, dtype=np.int16)
    g_app = _band(np.asarray(app.hs_gpa_recalc, dtype=np.float64), GPA_BANDS)
    g_alum = _band(np.asarray(alum.entry_hs_gpa_recalc, dtype=np.float64), GPA_BANDS)
    M[:, :, 0] = 0.5 * _match_ordinal(r_app, r_alum, 4.0) \
        + 0.5 * _match_ordinal(g_app, g_alum, float(len(GPA_BANDS)))

    # ---- d1 field of study: exact major, then same-school partial credit
    majors = sorted(set(app.intended_major) | set(alum.entry_intended_major))
    schools = sorted(set(app.intended_school) | set(alum.entry_intended_school))
    m_app = _codes(app.intended_major, majors)
    m_alum = _codes(alum.entry_intended_major, majors)
    s_app = _codes(app.intended_school, schools)
    s_alum = _codes(alum.entry_intended_school, schools)
    exact = _match_exact(m_app, m_alum)
    same_school = _match_exact(s_app, s_alum)
    M[:, :, 1] = np.maximum(exact, 0.5 * same_school)

    # ---- d2 activity portfolio, rarity weighted (SR-4)
    universe = ds.activities.activity_id.tolist()
    Sa, na = _weighted_sets(ds.app_ids, universe, ds.act_sets,
                            ds.act_weights, ds.act_idf)
    Se, ne = _weighted_sets(ds.alum_ids, universe, ds.act_sets,
                            ds.act_weights, ds.act_idf)
    M[:, :, 2], A[:, :, 2] = _tanimoto(Sa, na, Se, ne)

    # ---- d3 athletics
    M[:, :, 3], A[:, :, 3] = _athletics(ds)

    # ---- d4 leadership depth
    l_app = _leadership(ds.app_ids, ds)
    l_alum = _leadership(ds.alum_ids, ds)
    M[:, :, 4] = _match_ordinal(l_app, l_alum, 5.0)

    # ---- d5 school and geography
    regions = sorted(set(app.region) | set(alum.entry_region))
    hs = sorted(set(app.hs_type) | set(alum.entry_hs_type))
    M[:, :, 5] = 0.5 * _match_exact(_codes(app.region, regions),
                                    _codes(alum.entry_region, regions)) \
        + 0.5 * _match_exact(_codes(app.hs_type, hs),
                             _codes(alum.entry_hs_type, hs))

    # ---- d6 socioeconomic context (FR-2: off by default, not absent)
    fg = _match_exact(_codes(app.first_generation, ["N", "Y"]),
                      _codes(alum.entry_first_generation, ["N", "Y"]))
    inc = _match_ordinal(_codes(app.family_income_band, INCOME_ORDER),
                         _codes(alum.entry_family_income_band, INCOME_ORDER),
                         float(len(INCOME_ORDER) - 1))
    M[:, :, 6] = 0.5 * fg + 0.5 * inc

    # ---- d7 engagement
    M[:, :, 7] = _match_ordinal(
        np.asarray(app.demonstrated_interest_rating, dtype=np.int16),
        np.asarray(alum.entry_demonstrated_interest_rating, dtype=np.int16), 4.0)

    # ---- d8 campus club affinity (SR-7), proxy_sensitive clubs excluded
    from core.clubs import club_affinity
    M[:, :, 8], A[:, :, 8] = club_affinity(ds)

    return M, A


def flatten(M: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (M.reshape(-1, cfg.N_DIM), A.reshape(-1, cfg.N_DIM))


if __name__ == "__main__":
    import time
    from core.loader import load

    ds = load()
    t = time.perf_counter()
    M, A = build_tensors(ds)
    print(f"built in {(time.perf_counter() - t) * 1000:.0f} ms")
    print(f"M {M.shape} {M.nbytes / 1e6:.1f} MB   A {A.nbytes / 1e6:.1f} MB")
    print(f"{'d':>2} {'dimension':24s} {'mean':>6} {'max':>5} {'applicable':>11}")
    for d, name in enumerate(dimension_names()):
        ap = A[:, :, d]
        vals = M[:, :, d][ap > 0]
        print(f"{d:2d} {name:24s} {vals.mean():6.3f} {vals.max():5.2f} "
              f"{ap.mean():10.1%}")
