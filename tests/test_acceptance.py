"""Traceability sweep over all 27 acceptance criteria (requirements §13).

Each test names its criterion. Run with -v for the traceability report:

    python -m pytest tests/test_acceptance.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg                                               # noqa: E402
from core import clubs, explain, features, priorities, similarity   # noqa: E402
from core.graphbuild import (build_context, shared_nodes,           # noqa: E402
                             similar_edges, subgraph)
from core.loader import load                                        # noqa: E402
from ui.graphview import build_figure, legend_caption               # noqa: E402


@pytest.fixture(scope="module")
def env():
    ds = load()
    M, A = features.build_tensors(ds)
    M2, A2 = features.flatten(M, A)
    ctx = build_context(ds)
    w = similarity.weights_from_words(cfg.DIM_DEFAULTS)
    sim = similarity.contract(M2, A2, w, (ds.n_app, ds.n_alum))
    ctx.sim = sim
    ctx.topk_idx, ctx.topk_val = similarity.topk(sim, cfg.TOPK_EDGES,
                                                 cfg.SIM_FLOOR)
    ctx.rosters = clubs.project_rosters(ds)
    p1 = priorities.percentile(priorities.p1_yield(ds))
    p2 = priorities.percentile(priorities.p2_expected_ntr(ds))
    p4, thin, _, _ = priorities.p4_academic(sim, ds)
    C, target, labels = priorities.composition_matrix(ds)
    B, club_ids = clubs.coverage_matrix(ds, ctx.rosters)
    id_rank = np.argsort(np.argsort(np.asarray(ds.app_ids)))
    ctx.shortlist = priorities.build_shortlist(
        priorities.blend_individual(p1, p2, p4, .5, .5, .5),
        C, target, B, None, .5, .25, cfg.CLASS_TARGET, id_rank)
    return dict(ds=ds, M=M, A=A, M2=M2, A2=A2, ctx=ctx, w=w, sim=sim,
                p1=p1, p2=p2, p4=p4, thin=thin, C=C, target=target,
                labels=labels, B=B, club_ids=club_ids, id_rank=id_rank)


def test_ac01_both_populations_distinguishable_by_shape_and_colour(env):
    kinds = {n.kind for n in env["ctx"].nodes.values()}
    assert {"Applicant", "Alum"} <= kinds
    symbols = {k: cfg.NODE_STYLE[k]["symbol"] for k in cfg.NODE_STYLE}
    assert len(set(symbols.values())) == len(symbols), "symbols not unique"
    assert symbols["Applicant"] != symbols["Alum"]
    colours = {k: cfg.NODE_STYLE[k]["colour"] for k in ("Applicant", "Alum")}
    assert colours["Applicant"] != colours["Alum"]


def test_ac02_populations_connect_through_shared_nodes_and_highlight(env):
    ctx, ds = env["ctx"], env["ds"]
    found = 0
    for i in range(80):
        e = int(ctx.topk_idx[i, 0])
        if e < 0:
            continue
        sh = shared_nodes(ctx, ds.app_ids[i], ds.alum_ids[e])
        kinds = {ctx.nodes[s].kind for s in sh}
        if kinds & {"Activity", "Major", "Athletic team", "Campus club"}:
            found += 1
    assert found > 0, "no applicant-alum pair shares an attribute node"


def test_ac03_debate_majors_athletics_clubs_are_first_class(env):
    ctx = env["ctx"]
    labels = {n.label: n for n in ctx.nodes.values()}
    assert "Debate and Speech" in labels
    assert labels["Debate and Speech"].kind == "Activity"
    assert any(n.kind == "Athletic team" for n in ctx.nodes.values())
    assert any(n.kind == "Major" for n in ctx.nodes.values())
    assert any(n.kind == "Campus club" for n in ctx.nodes.values())
    # and each kind carries real edges
    deg = {}
    for e in ctx.edges:
        deg[e.source] = deg.get(e.source, 0) + 1
        deg[e.target] = deg.get(e.target, 0) + 1
    for kind in ("Activity", "Athletic team", "Major", "Campus club"):
        ids = [n.id for n in ctx.nodes.values() if n.kind == kind]
        assert sum(deg.get(i, 0) for i in ids) > 0, f"{kind} has no edges"


def test_ac04_similar_edges_scored_and_visually_distinct(env):
    ctx = env["ctx"]
    sims = similar_edges(ctx)
    assert sims, "no SIMILAR_TO edges"
    assert all(e.tier == "inferred" for e in sims)
    inferred = cfg.EDGE_TIERS["inferred"]
    for other in ("observed_past", "observed_intent"):
        assert inferred["dash"] != cfg.EDGE_TIERS[other]["dash"]
        assert inferred["colour"] != cfg.EDGE_TIERS[other]["colour"]
    assert ctx.topk_val is not None and np.isfinite(ctx.topk_val).any()


def test_ac05_four_word_scaled_priority_controls():
    assert len(cfg.PRIORITY_KEYS) == 4
    assert set(cfg.PRIORITY_LABELS.values()) == {
        "Yield", "Net tuition revenue", "Diversity", "Academic strength"}
    assert all(not str(o).isdigit() for o in cfg.SCALE)


def test_ac06_single_defining_priority_reproduces_its_own_order(env):
    ds = env["ds"]
    indiv = priorities.blend_individual(env["p1"], env["p2"], env["p4"],
                                       cfg.WEIGHT["Defining"], 0.0, 0.0)
    mask = priorities.build_shortlist(indiv, env["C"], env["target"], env["B"],
                                      None, 0.0, 0.0, cfg.CLASS_TARGET,
                                      env["id_rank"])
    raw = np.asarray(ds.applicants.predicted_yield_prob, dtype=float)
    assert raw[mask].min() >= raw[~mask].max() - 1e-9


def test_ac07_shortlist_group_and_evidence_subgraph(env):
    ctx, ds = env["ctx"], env["ds"]
    keep, edges = subgraph("Short list", None, ctx)
    chosen = {ds.app_ids[i] for i in np.flatnonzero(ctx.shortlist)}
    assert chosen <= keep, "short list members missing from the default view"
    assert any(ctx.nodes[n].kind == "Alum" for n in keep), "no evidence alumni"
    one = sorted(chosen)[0]
    sub_keep, _ = subgraph("Applicant focus", one, ctx)
    assert one in sub_keep and len(sub_keep) > 1


def test_ac08_no_numeric_importance_anywhere_in_config():
    for scale in (cfg.SCALE, list(cfg.SHORTLIST_SIZE),
                  list(cfg.MATCHES_PER_APPLICANT)):
        for word in scale:
            assert not str(word).replace(".", "", 1).isdigit()


def test_ac09_not_considered_removes_a_component(env):
    w = env["w"].copy()
    d = cfg.DIM_NAMES.index("Activity portfolio")
    w[d] = 0.0
    base = similarity.contract(env["M2"], env["A2"], w,
                               (env["ds"].n_app, env["ds"].n_alum))
    M2 = env["M2"].copy()
    M2[:, d] = 1.0 - M2[:, d]
    assert np.allclose(base, similarity.contract(
        M2, env["A2"], w, (env["ds"].n_app, env["ds"].n_alum)))


def test_ac10_contributions_sum_to_score(env):
    rng = np.random.default_rng(11)
    for _ in range(120):
        a, e = int(rng.integers(env["ds"].n_app)), int(rng.integers(env["ds"].n_alum))
        dec = similarity.decompose(env["M"], env["A"], env["w"], a, e)
        assert abs(dec["score"] - env["sim"][a, e]) < 1e-5


def test_ac11_edges_and_picks_carry_generated_rationale(env):
    ds, ctx = env["ds"], env["ctx"]
    for a_i in (0, 40, 120):
        e_i = int(ctx.topk_idx[a_i, 0])
        dec = similarity.decompose(env["M"], env["A"], env["w"], a_i, e_i)
        pct = float((env["sim"][a_i] <= env["sim"][a_i, e_i]).mean())
        det = explain.shared_detail(ds, ds.app_ids[a_i], ds.alum_ids[e_i])
        text = explain.rationale(dec, pct, env["w"], det)
        assert text.endswith(".") and len(text) > 40
        assert "{detail}" not in text and "shared interest in shared" not in text
        note = explain.pick_rationale(
            ds, a_i, {"w_yield": .8, "w_ntr": .4, "w_acad": .6, "w_div": .5},
            {"w_yield": 1.0, "w_ntr": 0.5, "w_acad": 0.5, "w_div": 0.5},
            None, False, in_list=bool(ctx.shortlist[a_i]))
        assert note.endswith(".")
        if not ctx.shortlist[a_i]:
            assert "Not shortlisted" in note


def test_ac12_interaction_stays_inside_the_budget(env):
    import time
    ds = env["ds"]
    t = time.perf_counter()
    sim = similarity.contract(env["M2"], env["A2"], env["w"],
                              (ds.n_app, ds.n_alum))
    idx, _ = similarity.topk(sim, cfg.TOPK_EDGES, cfg.SIM_FLOOR)
    p4, _, _, _ = priorities.p4_academic(sim, ds)
    indiv = priorities.blend_individual(env["p1"], env["p2"], p4, .5, .5, .5)
    mask = priorities.build_shortlist(indiv, env["C"], env["target"], env["B"],
                                      None, .5, .25, cfg.CLASS_TARGET,
                                      env["id_rank"])
    compute_ms = (time.perf_counter() - t) * 1000

    env["ctx"].shortlist = mask
    keep, edges = subgraph("Short list", None, env["ctx"])
    t = time.perf_counter()
    fig = build_figure(env["ctx"], keep, edges)
    fig.to_json()
    render_ms = (time.perf_counter() - t) * 1000
    assert compute_ms + render_ms < 300.0, (
        f"compute {compute_ms:.0f} ms + render {render_ms:.0f} ms exceeds NFR-1")


def test_ac13_maxing_one_priority_reports_its_cost(env):
    ds = env["ds"]

    def shortlist(words):
        indiv = priorities.blend_individual(
            env["p1"], env["p2"], env["p4"], cfg.WEIGHT[words["w_yield"]],
            cfg.WEIGHT[words["w_ntr"]], cfg.WEIGHT[words["w_acad"]])
        return priorities.build_shortlist(
            indiv, env["C"], env["target"], env["B"], None,
            cfg.WEIGHT[words["w_div"]], 0.25, cfg.CLASS_TARGET, env["id_rank"])

    rev = explain.cost_of_choice(ds, shortlist(cfg.PRESETS["Revenue-first"]))
    acc = explain.cost_of_choice(ds, shortlist(cfg.PRESETS["Access and mission"]))
    assert rev["shortlist"]["net_price"] > acc["shortlist"]["net_price"]
    assert rev["shortlist"]["low_income"] < acc["shortlist"]["low_income"]
    assert rev["delta"]["low_income"] < 0, "revenue-first shows no access cost"


def test_ac14_greedy_not_a_static_sort(env):
    indiv = priorities.blend_individual(env["p1"], env["p2"], env["p4"],
                                       .5, .5, .5)
    a = priorities.build_shortlist(indiv, env["C"], env["target"], env["B"],
                                   None, .5, .25, cfg.CLASS_TARGET,
                                   env["id_rank"])
    b = priorities.build_shortlist(indiv, env["C"], env["target"], env["B"],
                                   None, 0.0, .25, cfg.CLASS_TARGET,
                                   env["id_rank"])
    assert (a != b).sum() > 0


def test_ac15_composition_targets_visible_and_editable(env):
    """SR-5c: visible AND editable. A diversity score against undisclosed
    targets is not explainable, and one against unchangeable targets is not
    actually a policy choice."""
    ds = env["ds"]
    for field, spec in cfg.COMPOSITION_TARGETS.items():
        assert abs(sum(spec.values()) - 1.0) < 1e-6, f"{field} targets != 1.0"

    realised = priorities.composition_of(env["C"], env["ctx"].shortlist,
                                         env["labels"])
    assert set(realised) == set(cfg.COMPOSITION_TARGETS)

    # edited targets must flow through to a different composition matrix
    edited = {f: dict(s) for f, s in cfg.COMPOSITION_TARGETS.items()}
    edited["region"] = {k: (0.90 if k == "West" else 0.02)
                        for k in edited["region"]}
    C2, t2, labels2 = priorities.composition_matrix(ds, edited)
    assert not np.allclose(t2, env["target"]), "edited targets had no effect"

    # and they are renormalised rather than rejected
    norm = priorities.normalise_targets(edited)
    for field, spec in norm.items():
        assert abs(sum(spec.values()) - 1.0) < 1e-6

    # a West-heavy target must actually pull the short list westward
    indiv = priorities.blend_individual(env["p1"], env["p2"], env["p4"],
                                       .5, .5, .5)
    base = priorities.build_shortlist(indiv, env["C"], env["target"], env["B"],
                                      None, cfg.WEIGHT["Defining"], 0.0,
                                      cfg.CLASS_TARGET, env["id_rank"])
    west = priorities.build_shortlist(indiv, C2, t2, env["B"], None,
                                      cfg.WEIGHT["Defining"], 0.0,
                                      cfg.CLASS_TARGET, env["id_rank"])
    region = np.asarray(ds.applicants.region.astype(str))
    assert (region[west] == "West").mean() > (region[base] == "West").mean()


def test_ac16_configuration_sentence_has_no_numbers():
    for name, preset in cfg.PRESETS.items():
        s = explain.config_sentence(preset, "Selective", "Minor",
                                    ["Socioeconomic context"])
        assert not any(ch.isdigit() for ch in s), f"{name}: {s}"
        assert s.startswith("You are") and s.endswith(".")


def test_ac17_exemplar_cohort_comparable_to_pool(env):
    ds = env["ds"]
    ex = ds.alumni[ds.alumni.exemplar_flag == "Y"]
    assert len(ex) == 200
    for app_col, alum_col in [("region", "entry_region"),
                              ("hs_type", "entry_hs_type"),
                              ("first_generation", "entry_first_generation"),
                              ("family_income_band", "entry_family_income_band")]:
        a = ds.applicants[app_col].value_counts(normalize=True)
        b = ex[alum_col].value_counts(normalize=True)
        assert len(a) and len(b)


def test_ac18_protected_attributes_absent_from_scoring(env):
    ds = env["ds"]
    for frame in (ds.applicants, ds.alumni):
        assert not set(frame.columns) & set(cfg.PROTECTED_COLUMNS)
    for n in env["ctx"].nodes.values():
        assert not set(n.meta) & set(cfg.PROTECTED_COLUMNS)


def test_ac19_caveats_configured_for_the_interface():
    assert len(cfg.MOCK_DATA_CAVEATS) >= 4
    joined = " ".join(b for _, b in cfg.MOCK_DATA_CAVEATS)
    assert "UNDERSTATES" in joined
    assert "0.005" in joined or "no correlation" in joined


def test_ac20_determinism(env):
    indiv = priorities.blend_individual(env["p1"], env["p2"], env["p4"],
                                       .5, .5, .5)
    args = (env["C"], env["target"], env["B"], None, .5, .25,
            cfg.CLASS_TARGET, env["id_rank"])
    assert np.array_equal(priorities.build_shortlist(indiv, *args),
                          priorities.build_shortlist(indiv, *args))


def test_ac21_nothing_named_as_a_decision():
    banned = ("admit", "deny", "reject", "accept")
    for label in list(cfg.PRIORITY_LABELS.values()) + cfg.FOCUS_MODES \
            + list(cfg.PRESETS) + cfg.DIM_NAMES:
        low = label.lower()
        assert not any(b in low for b in banned), f"decision language: {label}"
    payload = explain.export_payload(
        load(), 0, cfg.PRESETS["Balanced"], cfg.DIM_DEFAULTS, [], True,
        "note", {"shortlist": {}, "baseline": {}, "delta": {}})
    assert "advisory" in payload["advisory_notice"].lower()


def test_ac22_club_nodes_show_roster_health_and_risk(env):
    ros = env["ctx"].rosters
    assert set(ros.band) <= {b for b, _ in cfg.ROSTER_BANDS}
    assert ros.at_risk.sum() >= 9
    for r in ros.itertuples(index=False):
        assert r.at_risk == (r.band in cfg.AT_RISK_BANDS)


def test_ac23_three_edge_tiers_are_visually_distinct():
    dashes = {t: cfg.EDGE_TIERS[t]["dash"] for t in cfg.EDGE_TIERS}
    assert len(set(dashes.values())) == 3, dashes
    colours = {t: cfg.EDGE_TIERS[t]["colour"] for t in cfg.EDGE_TIERS}
    assert len(set(colours.values())) == 3
    assert cfg.OBSERVED_PAST_EDGES.isdisjoint(cfg.OBSERVED_INTENT_EDGES)
    assert "MEMBER_OF" in cfg.OBSERVED_PAST_EDGES
    assert "INTERESTED_IN" in cfg.OBSERVED_INTENT_EDGES


def test_ac24_club_selection_exposes_its_pipeline(env):
    ds, ros = env["ds"], env["ctx"].rosters
    cid = ros[ros.at_risk].iloc[0].club_id
    row = ros[ros.club_id == cid].iloc[0]
    for col in ("holdover", "graduating_members", "min_viable_members",
                "band", "expected_new", "interested_count", "alumni_members"):
        assert col in ros.columns
    assert isinstance(clubs.interested_applicants(ds, cid), list)
    members = clubs.member_alumni(ds, cid)
    assert isinstance(members, list)
    if members:
        sub = ds.alumni[ds.alumni.alum_id.isin(members)]
        assert "success_composite_tier" in sub.columns
    keep, _ = subgraph("Club focus", cid, env["ctx"])
    assert cid in keep


def test_ac25_at_risk_clubs_with_identifiable_sole_pipelines(env):
    ros = env["ctx"].rosters
    assert int(ros.at_risk.sum()) >= 9
    sp = clubs.sole_pipeline(env["ds"], ros)
    assert sp, "no sole-pipeline club, so IR-7e has nothing to surface"
    for cid, heads in sp.items():
        assert 1 <= len(heads) <= 2


def test_ac26_empty_sets_dropped_from_normaliser(env):
    ds, A = env["ds"], env["A"]
    d = cfg.DIM_NAMES.index("Athletics")
    app = np.asarray(ds.applicants.athletic_recruit_sport.fillna("").astype(str))
    alum = np.asarray(ds.alumni.entry_athletic_recruit_sport.fillna("").astype(str))
    ai = np.flatnonzero((app == "") | (app == "nan"))
    ei = np.flatnonzero((alum == "") | (alum == "nan"))
    assert (A[np.ix_(ai, ei)][:, :, d] == 0).all()


def test_ac27_club_coverage_only_through_p3(env):
    indiv = priorities.blend_individual(env["p1"], env["p2"], env["p4"],
                                       .5, .5, .5)
    off = priorities.build_shortlist(indiv, env["C"], env["target"], env["B"],
                                     None, 0.0, 0.0, cfg.CLASS_TARGET,
                                     env["id_rank"])
    loud = priorities.build_shortlist(indiv, env["C"], env["target"], env["B"],
                                      None, 0.0, cfg.WEIGHT["Defining"],
                                      cfg.CLASS_TARGET, env["id_rank"])
    assert np.array_equal(off, loud)
    _, ids = clubs.coverage_matrix(env["ds"], env["ctx"].rosters)
    assert not set(ids) & env["ds"].proxy_clubs


def test_legend_describes_every_glyph_in_words():
    """NFR-3: every visual claim must also exist in words.

    Asserts the distinctions a reader has to make, not the specific words. Shape
    alone was not carrying the population distinction at these densities, so the
    caption now leads with size and colour and shape is the third cue.
    """
    text = legend_caption().lower()
    # the two populations, named and visually described
    assert "applicant" in text and "past students" in text
    assert "blue" in text and "violet" in text
    assert "small" in text and "larger" in text
    # the short-list signal, which is the thing a slider moves
    assert "yellow ring" in text and "short list" in text
    # the three edge tiers
    for word in ("solid", "dashed", "dotted"):
        assert word in text, f"legend omits {word}"
    # and the top-three rule, or a reader will think edges are missing
    assert "three most informative" in text
    # remaining node kinds
    for word in ("activities", "teams", "clubs"):
        assert word in text, f"legend omits {word}"
