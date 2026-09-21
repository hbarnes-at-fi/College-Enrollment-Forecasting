"""Headless correctness gate (specs §13). Every test names the criterion it covers.

Phase 2 must be green before any rendering work, because every correctness
acceptance criterion is testable without a browser.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg                                      # noqa: E402
from core import clubs, features, priorities, similarity   # noqa: E402
from core.loader import DataContractError, load, suppression_audit  # noqa: E402


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def ds():
    return load()


@pytest.fixture(scope="module")
def tensors(ds):
    M, A = features.build_tensors(ds)
    return M, A, *features.flatten(M, A)


@pytest.fixture(scope="module")
def ctx(ds, tensors):
    M, A, M2, A2 = tensors
    w = similarity.weights_from_words(cfg.DIM_DEFAULTS)
    sim = similarity.contract(M2, A2, w, (ds.n_app, ds.n_alum))
    C, target, labels = priorities.composition_matrix(ds)
    ros = clubs.project_rosters(ds)
    B, club_ids = clubs.coverage_matrix(ds, ros)
    id_rank = np.argsort(np.argsort(np.asarray(ds.app_ids)))
    return dict(M=M, A=A, M2=M2, A2=A2, w=w, sim=sim, C=C, target=target,
                labels=labels, ros=ros, B=B, club_ids=club_ids, id_rank=id_rank)


def _indiv(ds, ctx, wy, wn, wa):
    p1 = priorities.percentile(priorities.p1_yield(ds))
    p2 = priorities.percentile(priorities.p2_expected_ntr(ds))
    p4, _, _, _ = priorities.p4_academic(ctx["sim"], ds)
    return priorities.blend_individual(p1, p2, p4, wy, wn, wa)


# ------------------------------------------------------------------ AC-10
def test_contributions_sum_to_score(ds, ctx):
    """XR-1: decomposition is exact by construction, not by tolerance."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = int(rng.integers(ds.n_app))
        e = int(rng.integers(ds.n_alum))
        dec = similarity.decompose(ctx["M"], ctx["A"], ctx["w"], a, e)
        assert abs(dec["score"] - ctx["sim"][a, e]) < 1e-5, (
            f"pair ({a},{e}) contributions {dec['score']} != "
            f"score {ctx['sim'][a, e]}")


# ------------------------------------------------------------------- AC-9
def test_not_considered_is_inert(ds, ctx):
    """IR-1b: a zero weight must cancel from numerator and denominator together,
    leaving every score untouched by that dimension's values."""
    for d, name in enumerate(cfg.DIM_NAMES):
        w_off = ctx["w"].copy()
        w_off[d] = 0.0
        base = similarity.contract(ctx["M2"], ctx["A2"], w_off,
                                   (ds.n_app, ds.n_alum))

        # corrupt that dimension entirely; scores must not move
        M2 = ctx["M2"].copy()
        M2[:, d] = 1.0 - M2[:, d]
        after = similarity.contract(M2, ctx["A2"], w_off, (ds.n_app, ds.n_alum))
        assert np.allclose(base, after), f"dimension {name} leaked at zero weight"

        dec = similarity.decompose(ctx["M"], ctx["A"], w_off, 0, 0)
        assert name not in dec["contributions"]


# ------------------------------------------------------------------- AC-6
@pytest.mark.parametrize("which,col,ascending", [
    ("yield", "predicted_yield_prob", False),
    ("acad", None, False),
])
def test_defining_reproduces_single_priority_order(ds, ctx, which, col, ascending):
    """One priority at Defining and the rest at Not considered must reproduce
    that priority's own ranking. Percentile rank is monotone, so this holds
    exactly for the three individual priorities."""
    W = cfg.WEIGHT
    if which == "yield":
        indiv = _indiv(ds, ctx, W["Defining"], 0.0, 0.0)
        raw = np.asarray(ds.applicants[col], dtype=np.float64)
    else:
        indiv = _indiv(ds, ctx, 0.0, 0.0, W["Defining"])
        raw, _, _, _ = priorities.p4_academic(ctx["sim"], ds)
        raw = np.asarray(raw, dtype=np.float64)

    n = cfg.CLASS_TARGET
    mask = priorities.build_shortlist(indiv, ctx["C"], ctx["target"], ctx["B"],
                                      None, 0.0, 0.0, n, ctx["id_rank"])
    assert mask.sum() == n
    chosen = np.flatnonzero(mask)
    rest = np.flatnonzero(~mask)
    assert raw[chosen].min() >= raw[rest].max() - 1e-9, (
        f"{which}: short list is not the top {n} by its own priority")


def test_expected_ntr_is_not_nominal(ds):
    """SR-6: a full-pay applicant who will not enrol contributes nothing."""
    net = np.asarray(ds.applicants.net_price, dtype=np.float64)
    prob = np.asarray(ds.applicants.predicted_yield_prob, dtype=np.float64)
    ntr = priorities.p2_expected_ntr(ds)
    assert np.allclose(ntr, net * prob)
    # and the two orderings genuinely differ, else the distinction is cosmetic
    assert not np.array_equal(np.argsort(net), np.argsort(ntr))


# ------------------------------------------------------------------- AC-6
def test_percentile_is_monotone(ds):
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    p = priorities.percentile(x)
    assert np.array_equal(np.argsort(np.argsort(x)), np.argsort(np.argsort(p)))
    assert p.min() >= 0.0 and p.max() <= 1.0


# ------------------------------------------------------------------ AC-14
def test_greedy_differs_from_static_sort(ds, ctx):
    """SR-5b: sorting on a precomputed diversity column is wrong, because each
    pick changes the value of every remaining candidate."""
    indiv = _indiv(ds, ctx, 0.5, 0.5, 0.5)
    greedy = priorities.build_shortlist(indiv, ctx["C"], ctx["target"], ctx["B"],
                                        None, 0.5, 0.25, cfg.CLASS_TARGET,
                                        ctx["id_rank"])
    sorted_only = priorities.build_shortlist(indiv, ctx["C"], ctx["target"],
                                             ctx["B"], None, 0.0, 0.25,
                                             cfg.CLASS_TARGET, ctx["id_rank"])
    assert greedy.sum() == sorted_only.sum() == cfg.CLASS_TARGET
    assert (greedy != sorted_only).sum() > 0


def test_greedy_improves_composition(ds, ctx):
    """P3 must actually move the class toward targets, not merely differ."""
    indiv = _indiv(ds, ctx, 0.5, 0.5, 0.5)

    def distance(mask):
        n = int(mask.sum())
        return float(np.abs(ctx["C"][mask].sum(axis=0) / n - ctx["target"]).sum())

    off = priorities.build_shortlist(indiv, ctx["C"], ctx["target"], ctx["B"],
                                     None, 0.0, 0.0, cfg.CLASS_TARGET,
                                     ctx["id_rank"])
    on = priorities.build_shortlist(indiv, ctx["C"], ctx["target"], ctx["B"],
                                    None, cfg.WEIGHT["Defining"], 0.0,
                                    cfg.CLASS_TARGET, ctx["id_rank"])
    assert distance(on) < distance(off)


# ------------------------------------------------------------------ AC-20
def test_determinism(ds, ctx):
    """NFR-5: identical configuration yields identical output, including the
    tie-breaks, which np.argmax would resolve by CSV row order."""
    indiv = _indiv(ds, ctx, 0.5, 0.5, 0.5)
    args = (ctx["C"], ctx["target"], ctx["B"], None, 0.5, 0.25,
            cfg.CLASS_TARGET, ctx["id_rank"])
    a = priorities.build_shortlist(indiv, *args)
    b = priorities.build_shortlist(indiv, *args)
    assert np.array_equal(a, b)


def test_tie_breaks_ignore_row_order(ds, ctx):
    """A flat score must select by applicant_id, not by position."""
    flat = np.zeros(ds.n_app, dtype=np.float32)
    picked = priorities.stable_top_n(flat, 10, ctx["id_rank"])
    expected = set(np.argsort(np.asarray(ds.app_ids))[:10].tolist())
    assert set(np.flatnonzero(picked).tolist()) == expected


# ------------------------------------------------------------------ AC-18
def test_no_protected_attributes(ds):
    """SR-3 / FR-5: dropped at the boundary, so exclusion is one assertion over
    Dataset rather than an inspection of every code path."""
    for frame in (ds.applicants, ds.alumni):
        assert not (set(frame.columns) & set(cfg.PROTECTED_COLUMNS))
    assert all(suppression_audit(ds).values())
    assert not (set(cfg.DIM_NAMES) & {"Ethnicity", "Gender", "Race", "Sex"})


def test_no_streamlit_in_core():
    """specs §5: core/ and data/ stay importable without Streamlit, enforced by
    parsing imports rather than by convention."""
    offenders = []
    for folder in ("core", "data"):
        for path in (ROOT / folder).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(n.split(".")[0] == "streamlit" for n in names):
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"streamlit imported in {offenders}"


# ------------------------------------------------------------------ AC-26
def test_empty_sets_are_dropped_not_scored(ds, tensors):
    """SR-7b: absence is not agreement. Two non-athletes must not match on
    athletics, and two club-less people must not match on clubs."""
    M, A, _, _ = tensors

    d_ath = cfg.DIM_NAMES.index("Athletics")
    app_sport = np.asarray(ds.applicants.athletic_recruit_sport.fillna("").astype(str))
    alum_sport = np.asarray(ds.alumni.entry_athletic_recruit_sport.fillna("").astype(str))
    a_none = np.flatnonzero((app_sport == "") | (app_sport == "nan"))
    e_none = np.flatnonzero((alum_sport == "") | (alum_sport == "nan"))
    assert a_none.size and e_none.size
    block = A[np.ix_(a_none, e_none)][:, :, d_ath]
    assert (block == 0).all(), "mutual non-participation scored as applicable"

    d_club = cfg.DIM_NAMES.index("Campus club affinity")
    non_proxy = set(ds.clubs.club_id) - ds.proxy_clubs
    a_noclub = np.flatnonzero([
        not (ds.club_sets.get(p, set()) & non_proxy) for p in ds.app_ids])
    e_noclub = np.flatnonzero([
        not (ds.club_sets.get(p, set()) & non_proxy) for p in ds.alum_ids])
    if a_noclub.size and e_noclub.size:
        block = A[np.ix_(a_noclub, e_noclub)][:, :, d_club]
        assert (block == 0).all(), "two club-less people scored as comparable"


def test_proxy_clubs_excluded_from_similarity(ds, tensors):
    """§0.1: interest in an affinity or religious club is a protected-attribute
    proxy and must not reach the similarity dimension."""
    M, A, _, _ = tensors
    d = cfg.DIM_NAMES.index("Campus club affinity")
    proxy = ds.proxy_clubs
    assert proxy, "fixture has no proxy_sensitive clubs to test"

    # applicants and alumni whose ONLY clubs are proxy_sensitive must be
    # uncomparable on this dimension
    a_only = [i for i, p in enumerate(ds.app_ids)
              if ds.club_sets.get(p) and ds.club_sets[p] <= proxy]
    e_only = [i for i, p in enumerate(ds.alum_ids)
              if ds.club_sets.get(p) and ds.club_sets[p] <= proxy]
    if a_only and e_only:
        block = A[np.ix_(a_only, e_only)][:, :, d]
        assert (block == 0).all(), "proxy-only club sets leaked into scoring"


# ------------------------------------------------------------------ AC-25
def test_at_risk_clubs_and_sole_pipelines_exist(ds, ctx):
    """DR-6c and SR-8b: the club layer needs something to demonstrate."""
    ros = ctx["ros"]
    at_risk = ros[ros.at_risk]
    assert len(at_risk) >= 9, f"DR-6c wants >= 9 at-risk clubs, got {len(at_risk)}"
    sp = clubs.sole_pipeline(ds, ros)
    assert sp, "no at-risk club has a 1-2 person pipeline; IR-7e has nothing"
    for cid, heads in sp.items():
        assert 1 <= len(heads) <= 2
        assert bool(ros.loc[ros.club_id == cid, "at_risk"].iloc[0])


def test_at_risk_definition_matches_bands(ctx):
    """SR-8a and the config bands must agree on what 'at risk' means."""
    ros = ctx["ros"]
    for r in ros.itertuples(index=False):
        assert r.at_risk == (r.band in cfg.AT_RISK_BANDS), (
            f"{r.club_id}: at_risk={r.at_risk} but band={r.band}")


# ------------------------------------------------------------------ AC-27
def test_club_coverage_inert_when_p3_off(ds, ctx):
    """SR-8c: club coverage reaches the short list only through P3."""
    indiv = _indiv(ds, ctx, 0.5, 0.5, 0.5)
    a = priorities.build_shortlist(indiv, ctx["C"], ctx["target"], ctx["B"],
                                   None, 0.0, 0.0, cfg.CLASS_TARGET,
                                   ctx["id_rank"])
    b = priorities.build_shortlist(indiv, ctx["C"], ctx["target"], ctx["B"],
                                   None, 0.0, cfg.WEIGHT["Defining"],
                                   cfg.CLASS_TARGET, ctx["id_rank"])
    assert np.array_equal(a, b), "club weight moved the list with P3 off"


def test_proxy_clubs_excluded_from_coverage(ds, ctx):
    """§0.1: coverage credit would reward admitting by inferred identity."""
    _, club_ids = clubs.coverage_matrix(ds, ctx["ros"])
    assert not (set(club_ids) & ds.proxy_clubs)
    # and at least one proxy club IS at risk, so the exclusion is load-bearing
    ros = ctx["ros"]
    assert bool((ros.at_risk & ros.proxy_sensitive).any()), (
        "no at-risk proxy club in the fixture, so this test proves nothing")


def test_club_coverage_increases_with_p3(ds, ctx):
    indiv = _indiv(ds, ctx, 0.5, 0.5, 0.5)
    B = ctx["B"]

    def covered(w3, wc):
        m = priorities.build_shortlist(indiv, ctx["C"], ctx["target"], B, None,
                                       w3, wc, cfg.CLASS_TARGET, ctx["id_rank"])
        return int((B[m].sum(axis=0) > 0).sum())

    assert covered(cfg.WEIGHT["Moderate"], 1.0) > covered(0.0, 0.0)


# ------------------------------------------------- design 5.5, thin kNN
def test_thin_neighbourhood_falls_back_to_credentials(ds, ctx):
    """A confident academic score from no comparable alumni is the most
    dangerous failure available here, so the guard must actually fire."""
    # collapse similarity: weight only Athletics, which is inapplicable for the
    # ~85% of pairs where neither participated
    words = {n: "Not considered" for n in cfg.DIM_NAMES}
    words["Athletics"] = "Defining"
    w = similarity.weights_from_words(words)
    sim = similarity.contract(ctx["M2"], ctx["A2"], w, (ds.n_app, ds.n_alum))

    score, thin, _, parts = priorities.p4_academic(sim, ds)
    assert thin.sum() > 0, "guard never fires, so it is untested in practice"

    # Substantive check: for a thin applicant the validated component was
    # withheld, so their score cannot depend on which success components are
    # toggled. For a non-thin applicant it must.
    a = priorities.p4_academic(sim, ds, {"Attainment": True})[3]["raw"]
    b = priorities.p4_academic(sim, ds, {"Growth": True})[3]["raw"]
    assert np.allclose(a[thin], b[thin]), (
        "thin applicants moved with the success label despite validation "
        "being withheld")
    assert not np.allclose(a[~thin], b[~thin]), (
        "non-thin applicants ignored the success label, so validation is "
        "not actually wired in")
    # and the withheld group really is scored on credentials alone
    assert np.allclose(a[thin], parts["credential"][thin])


def test_success_components_stored_separately(ds):
    """DR-4c / requirements 1.4: the four definitions select different people,
    so they must not be collapsed into one hidden number."""
    cols = list(priorities.SUCCESS_COMPONENTS.values())
    for c in cols:
        assert c in ds.alumni.columns
    only_attain = priorities.success_scores(ds, {"Attainment": True})
    only_growth = priorities.success_scores(ds, {"Growth": True})
    assert not np.allclose(only_attain, only_growth)
    top_a = set(np.argsort(-only_attain)[:100].tolist())
    top_g = set(np.argsort(-only_growth)[:100].tolist())
    assert len(top_a & top_g) < 90, "components are effectively the same metric"


# ------------------------------------------------------- loader contracts
def test_validate_rejects_relationship_mismatch(ds):
    import copy
    bad = copy.copy(ds)
    bad.club_edges = ds.club_edges.copy()
    bad.club_edges.loc[bad.club_edges.index[0], "person_type"] = "Alum"
    bad.club_edges.loc[bad.club_edges.index[0], "relationship"] = "INTERESTED_IN"
    with pytest.raises(DataContractError):
        from core.loader import validate
        validate(bad)


def test_validate_rejects_null_discipline_breach(ds):
    import copy
    from core.loader import validate
    bad = copy.copy(ds)
    bad.club_edges = ds.club_edges.copy()
    row = bad.club_edges.index[
        (bad.club_edges.relationship == "INTERESTED_IN").to_numpy().argmax()]
    bad.club_edges.loc[row, "club_role"] = "President"
    with pytest.raises(DataContractError):
        validate(bad)


def test_tensor_shapes_and_range(ds, tensors):
    M, A, M2, A2 = tensors
    assert M.shape == (ds.n_app, ds.n_alum, cfg.N_DIM)
    assert A.shape == M.shape
    assert M.min() >= 0.0 and M.max() <= 1.0
    assert set(np.unique(A)) <= {0.0, 1.0}
    assert M2.shape == (ds.n_app * ds.n_alum, cfg.N_DIM)


def test_topk_respects_floor(ds, ctx):
    """NR-3b: a weak top match is not a match."""
    idx, val = similarity.topk(ctx["sim"], 5, 0.95)
    kept = idx >= 0
    assert np.all(val[kept] >= 0.95)
    assert kept.sum() < idx.size, "floor of 0.95 suppressed nothing"
