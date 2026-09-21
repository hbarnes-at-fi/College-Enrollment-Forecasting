"""Acceptance criteria 28 to 41: cohort clusters and progressive disclosure."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg                                              # noqa: E402
from core import cohorts, features, priorities, similarity        # noqa: E402
from core.graphbuild import (build_context, cluster_layout,        # noqa: E402
                             freeze_layout, similar_edges, subgraph)
from core.loader import load                                      # noqa: E402
from ui.graphview import (_filter_edges, build_figure,             # noqa: E402
                         detail_caption)


@pytest.fixture(scope="module")
def env():
    ds = load()
    ctx = build_context(ds)
    M, A = features.build_tensors(ds)
    M2, A2 = features.flatten(M, A)
    w = similarity.weights_from_words(cfg.DIM_DEFAULTS)
    sim = similarity.contract(M2, A2, w, (ds.n_app, ds.n_alum))
    ctx.sim = sim
    ctx.topk_idx, ctx.topk_val = similarity.topk(sim, cfg.TOPK_EDGES,
                                                 cfg.SIM_FLOOR)
    from core.clubs import coverage_matrix, project_rosters
    ctx.rosters = project_rosters(ds)
    B, _ = coverage_matrix(ds, ctx.rosters)
    C, target, labels = priorities.composition_matrix(ds)
    p1 = priorities.percentile(priorities.p1_yield(ds))
    p2 = priorities.percentile(priorities.p2_expected_ntr(ds))
    p4 = priorities.p4_academic(sim, ds)[0]
    ctx.shortlist = priorities.build_shortlist(
        priorities.blend_individual(p1, p2, p4, .5, .5, .5),
        C, target, B, None, .5, .25, cfg.CLASS_TARGET,
        np.argsort(np.argsort(np.asarray(ds.app_ids))))
    return dict(ds=ds, ctx=ctx, a=ctx.assignment)


# ------------------------------------------------------------------ AC-28
def test_ac28_named_clusters_with_geometry(env):
    ctx, a = env["ctx"], env["a"]
    assert ctx.clustered
    assert len(a.members) >= 8, "too few clusters to read as structure"
    for cohort in a.members:
        assert cohort in ctx.cluster_geo
        g = ctx.cluster_geo[cohort]
        assert cfg.CLUSTER_DISC_MIN <= g["radius"] <= cfg.CLUSTER_DISC_MAX
        # ring radius is derived from disc packing, not a constant
        assert g["ring"] >= cfg.CLUSTER_RING_RADIUS
        assert abs(math.hypot(*g["centre"]) - g["ring"]) < 1e-6

    # clusters must not overlap, or they stop reading as distinct groups
    items = list(ctx.cluster_geo.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ca, ga), (cb, gb) = items[i], items[j]
            gap = math.dist(ga["centre"], gb["centre"]) - (ga["radius"]
                                                           + gb["radius"])
            assert gap > 0, f"{ca} overlaps {cb}"


def test_ac28_attribute_nodes_stay_in_the_central_clearing(env):
    ctx = env["ctx"]
    attr = [n.id for n in ctx.nodes.values()
            if n.kind in ("Activity", "Athletic team", "Major", "Campus club")]
    worst = max(math.hypot(*ctx.layout[i]) for i in attr)
    assert worst <= cfg.CLUSTER_CENTRE_CLEAR + 1e-6, (
        f"attribute node at radius {worst:.3f} intrudes on the cluster ring")


# ------------------------------------------------------------ AC-29, AC-31
def test_ac29_every_cohort_holds_both_populations(env):
    ds, a = env["ds"], env["a"]
    for cohort in a.members:
        app, alum = a.split(ds, cohort)
        assert app and alum, (
            f"{cohort} is single-population, which breaks the NR-4a comparison "
            f"clustering exists to enable")


def test_ac29_populations_separate_within_a_cluster(env):
    """NR-4c: clustering must not cost the population distinction."""
    ds, ctx, a = env["ds"], env["ctx"], env["a"]
    for cohort in list(a.members)[:4]:
        app, alum = a.split(ds, cohort)
        centre = ctx.cluster_geo[cohort]["centre"]

        def mean_pos(ids):
            pts = [ctx.layout[i] for i in ids]
            return (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))

        pa, pl = mean_pos(app), mean_pos(alum)
        assert math.dist(pa, pl) > 0.05, (
            f"{cohort}: applicant and alumni lobes coincide")
        assert math.dist(pa, centre) < ctx.cluster_geo[cohort]["radius"]


def test_ac31_no_cohort_below_the_minimum(env):
    a = env["a"]
    for cohort, ids in a.members.items():
        if cohort == cohorts.GENERAL:
            continue
        assert len(ids) >= cohorts.MIN_COHORT_SIZE, (
            f"{cohort} has {len(ids)}, below the NR-4e floor")


# ------------------------------------------------------------------ AC-30
def test_ac30_assignment_is_total_and_single_valued(env):
    ds, a = env["ds"], env["a"]
    assert len(a.primary) == ds.n_app + ds.n_alum
    assigned = [p for ids in a.members.values() for p in ids]
    assert len(assigned) == len(set(assigned)) == ds.n_app + ds.n_alum


def test_ac30_memberships_retained_and_lead_with_primary(env):
    a = env["a"]
    multi = [p for p, m in a.memberships.items() if len(m) > 1]
    assert multi, "nobody matches several cohorts, so precedence is untested"
    for pid in multi[:200]:
        assert a.memberships[pid][0] == a.primary[pid], (
            "memberships order disagrees with the primary assignment")


def test_ac30_precedence_is_rarity_ordered(env):
    """NR-4g. The hand-picked order was disproved by the data; this is the rule
    that replaced it."""
    ds, a = env["ds"], env["a"]
    profiles = cohorts.build_profiles(ds)
    breadth = {d.name: sum(1 for p in profiles if d.predicate(p))
               for d in cohorts.COHORT_DEFS if d.name != cohorts.GENERAL}
    ranked = [c for c in a.order if c != cohorts.GENERAL]
    widths = [breadth[c] for c in ranked]
    assert widths == sorted(widths), f"precedence is not rarity-ordered: {widths}"
    assert a.order[-1] == cohorts.GENERAL


def test_ac30_access_cohort_survives_precedence(env):
    """The regression that motivated NR-4g: a hand-ordered precedence left only
    10 of 64 matching people in the access cohort, and it then vanished under the
    NR-4e floor, breaching FR-7's requirement that it ship visible."""
    a = env["a"]
    name = "Access and first-generation"
    assert name not in a.merged, "access cohort was merged away again"
    assert len(a.members.get(name, [])) >= cohorts.MIN_COHORT_SIZE


# ------------------------------------------------------------ AC-32, AC-33
@pytest.mark.parametrize("level,expect_edges,expect_similar", [
    ("Clusters only", False, False),
    ("Outlines", True, False),
    ("Connections", True, False),     # focused=False, so still no SIMILAR_TO
    ("Everything", True, True),
])
def test_ac32_detail_levels_gate_edges(env, level, expect_edges, expect_similar):
    ctx = env["ctx"]
    keep, edges = subgraph("Short list", None, ctx)
    assert any(e.kind == "SIMILAR_TO" for e in edges), "fixture has no inference"
    assert any(e.kind != "SIMILAR_TO" for e in edges), "fixture has no observed edges"

    spec = cfg.DETAIL_LEVELS[level]
    drawn = _filter_edges(edges, spec, focused=False, emphasis=None)
    assert bool(drawn) == expect_edges
    assert any(e.kind == "SIMILAR_TO" for e in drawn) == expect_similar


def test_ac33_similar_edges_appear_on_focus_only(env):
    ctx = env["ctx"]
    _, edges = subgraph("Short list", None, ctx)
    spec = cfg.DETAIL_LEVELS["Connections"]
    assert not any(e.kind == "SIMILAR_TO"
                   for e in _filter_edges(edges, spec, False, None))
    assert any(e.kind == "SIMILAR_TO"
               for e in _filter_edges(edges, spec, True, None))


def test_ac32_opacity_rises_with_detail():
    ops = [cfg.DETAIL_LEVELS[l]["edge_opacity"] for l in cfg.DETAIL_ORDER]
    assert ops == sorted(ops), f"detail levels are not monotone: {ops}"
    assert ops[0] == 0.0 and ops[-1] > 0.5


def test_ac32_clusters_only_emits_no_line_traces(env):
    ctx = env["ctx"]
    keep, edges = subgraph("Short list", None, ctx)
    fig = build_figure(ctx, keep, edges, detail="Clusters only")
    lines = [t for t in fig.data if t.mode == "lines"]
    assert not lines, "Clusters only drew edge traces"
    assert [t for t in fig.data if t.mode == "markers"], "no nodes drawn"
    assert fig.layout.shapes, "no cluster discs drawn"


# ------------------------------------------------------------------ AC-34
def test_ac34_highlight_overrides_every_detail_level(env):
    """NR-5d: an explanation that disappears because of a display setting is not
    an explanation."""
    ds, ctx = env["ds"], env["ctx"]
    a_id = ds.app_ids[0]
    e_id = ds.alum_ids[int(ctx.topk_idx[0, 0])]
    from core.graphbuild import shared_nodes
    sh = shared_nodes(ctx, a_id, e_id)
    emphasis = ({(a_id, s) for s in sh} | {(e_id, s) for s in sh}
                | {(a_id, e_id)})
    keep, edges = subgraph("Applicant focus", a_id, ctx)

    for level in cfg.DETAIL_ORDER:
        spec = cfg.DETAIL_LEVELS[level]
        drawn = _filter_edges(edges, spec, focused=False, emphasis=emphasis)
        assert drawn, f"{level}: highlight was suppressed entirely"
        fig = build_figure(ctx, keep, edges, highlight=sh | {a_id, e_id},
                           emphasis=emphasis, detail=level)
        assert any(t.mode == "lines" for t in fig.data), (
            f"{level}: no highlighted edge drawn")


# ------------------------------------------------------------------ AC-35
def test_ac35_detail_control_is_word_scaled_and_never_claims_zoom():
    for word in cfg.DETAIL_ORDER:
        assert not str(word).replace(".", "", 1).isdigit()
    joined = " ".join(cfg.DETAIL_DESCRIPTIONS.values()).lower()
    # describing what the user can do is fine; claiming the graph reacts to zoom
    # is not (NR-5e)
    for banned in ("as you zoom the graph responds",
                   "automatically", "detects zoom"):
        assert banned not in joined
    assert "pan and zoom in and they resolve" in joined, (
        "the opacity mechanism is not explained to the user")


# ------------------------------------------------------------------ AC-36
def test_ac36_clustering_is_removable(env):
    ctx = env["ctx"]
    # one layout per grouping, plus the arc fallback
    assert set(ctx.layouts) == set(cfg.GROUPINGS) | {"arcs"}
    clustered = dict(ctx.layout)
    ctx.set_arrangement(False)
    assert not ctx.clustered
    assert ctx.layout != clustered
    fig = build_figure(ctx, *subgraph("Short list", None, ctx))
    assert not fig.layout.shapes, "cluster discs drawn with clustering off"
    ctx.set_arrangement(True)
    assert ctx.layout == clustered, "switching back did not restore positions"


# ------------------------------------------------------------------ AC-37
def test_ac37_every_definition_and_the_order_are_disclosable(env):
    a = env["a"]
    defs = {d.name: d for d in cohorts.definitions()}
    for cohort in a.members:
        assert cohort in defs
        assert len(defs[cohort].description) > 20
    for name in ("Pre-law and civic", "Research-track sciences"):
        assert "conjunction" in defs[name].description.lower(), (
            f"{name} is a conjunction and must say so")
    assert a.order and len(a.order) == len(cohorts.COHORT_NAMES)


# ------------------------------------------------------------------ AC-38
def test_ac38_detail_level_is_stated_in_words():
    from core.explain import config_sentence
    s = config_sentence(cfg.PRESETS["Balanced"], "Selective", "Minor", [],
                        detail="Clusters only", clustered=True)
    assert "clusters only" in s.lower()
    assert "cohort clusters" in s.lower()
    assert not any(ch.isdigit() for ch in s)
    for level in cfg.DETAIL_ORDER:
        cap = detail_caption(level, True)
        assert len(cap) > 20


# ------------------------------------------------------------------ AC-39
def test_ac39_no_cohort_can_see_a_protected_attribute():
    """FR-7 is structural: a predicate cannot reference what Profile does not
    carry, and Profile cannot carry what Dataset dropped at load."""
    fields = set(cohorts.Profile.__dataclass_fields__)
    assert not fields & set(cfg.PROTECTED_COLUMNS)
    for banned in ("ethnic", "race", "gender", "sex", "religion"):
        assert not any(banned in f.lower() for f in fields)


def test_ac39_club_membership_cannot_reach_a_cohort(env):
    """§0.1: a cluster built on affinity-club interest is the same segregation
    map by another name."""
    ds = env["ds"]
    club_ids = set(ds.clubs.club_id)
    for p in cohorts.build_profiles(ds):
        assert not (p.activities & club_ids), (
            f"{p.pid} carries club ids in Profile.activities")


def test_ac39_access_cohort_is_removable(env):
    """FR-7: it ships visible AND removable, because a visible cluster is a
    stronger act than a scoring weight."""
    ctx = env["ctx"]
    ctx.set_arrangement(False)
    fig = build_figure(ctx, *subgraph("Short list", None, ctx))
    assert not fig.layout.annotations, "cohort labels survive with clustering off"
    ctx.set_arrangement(True)


# ------------------------------------------------------------------ AC-40
def test_ac40_clusters_encode_no_ranking(env):
    ctx = env["ctx"]
    keep, edges = subgraph("Short list", None, ctx)
    fig = build_figure(ctx, keep, edges, detail="Outlines")
    fills = {s.fillcolor for s in fig.layout.shapes}
    assert len(fills) == 1, f"cluster fills vary, implying a ranking: {fills}"
    assert fills == {cfg.CLUSTER_FILL}

    # a ring has no beginning, so ring position cannot encode an order
    angles = sorted(g["angle"] for g in ctx.cluster_geo.values())
    span = max(angles) - min(angles)
    assert span > math.pi, "clusters occupy an arc, not a ring, implying order"


# ------------------------------------------------------------------ AC-41
def test_ac41_both_arrangements_are_deterministic(env):
    ds, ctx, a = env["ds"], env["ctx"], env["a"]
    c1, g1 = cluster_layout(ctx.nodes, ds, a)
    c2, g2 = cluster_layout(ctx.nodes, ds, a)
    assert c1 == c2 and g1 == g2
    assert freeze_layout(ctx.nodes, ds) == freeze_layout(ctx.nodes, ds)


def test_ac41_assignment_depends_only_on_data(env):
    """NFR-6: if cohort assignment were weight-dependent, every node would move
    on every interaction and XR-4 would be defeated."""
    ds = env["ds"]
    a1, a2 = cohorts.assign(ds), cohorts.assign(ds)
    assert a1.primary == a2.primary
    assert a1.order == a2.order
    assert a1.members == a2.members


def test_cohort_isolate_focus_mode(env):
    ctx, a = env["ctx"], env["a"]
    cohort = a.order[0]
    keep, edges = subgraph("Cohort isolate", cohort, ctx)
    members = set(a.members[cohort])
    assert members <= keep
    people = {n for n in keep if ctx.nodes[n].kind in ("Applicant", "Alum")}
    assert people == members, "isolate leaked people from other cohorts"
    # nothing but people is drawn now, so the isolate IS its members
    assert keep == members


# ------------------------------------- short-list visibility and theming
def test_shortlist_gets_a_yellow_ring_and_its_own_legend_entry(env):
    """The short list is what a slider moves, so it needs the loudest signal
    available. A fill change across 755 nodes is invisible."""
    ctx = env["ctx"]
    # Cohort overview is the default precisely because it holds both groups; a
    # short-list-only view cannot show what a slider removed.
    keep, edges = subgraph("Cohort overview", None, ctx)
    fig = build_figure(ctx, keep, edges, detail="Outlines")

    names = {t.name for t in fig.data}
    assert "Applicant, on short list" in names, (
        "short list has no legend entry, leaving a yellow ring unexplained")
    assert "Applicant, not on short list" in names

    on = next(t for t in fig.data if t.name == "Applicant, on short list")
    off = next(t for t in fig.data if t.name == "Applicant, not on short list")
    assert set(np.atleast_1d(on.marker.line.color)) == {cfg.SHORTLIST_OUTLINE}
    assert float(np.min(on.marker.line.width)) >= cfg.SHORTLIST_OUTLINE_WIDTH
    assert off.opacity < on.opacity, "unpicked applicants are not pushed back"
    assert cfg.SHORTLIST_OUTLINE not in set(np.atleast_1d(off.marker.line.color))


def test_shortlist_membership_matches_the_mask(env):
    ds, ctx = env["ds"], env["ctx"]
    keep, edges = subgraph("Short list", None, ctx)
    fig = build_figure(ctx, keep, edges, detail="Outlines")
    on = next(t for t in fig.data if t.name == "Applicant, on short list")
    drawn = set(on.customdata)
    expected = {ds.app_ids[i] for i in np.flatnonzero(ctx.shortlist)} & keep
    assert drawn == expected


def test_cluster_labels_report_slider_effect(env):
    """The requested behaviour: a label must say how many of this group made the
    short list, and how that changed."""
    ds, ctx, a = env["ds"], env["ctx"], env["a"]
    keep, edges = subgraph("Short list", None, ctx)

    stats = {}
    for cohort, ids in a.members.items():
        idx = [ds.app_index[p] for p in ids if p in ds.app_index]
        if idx:
            stats[cohort] = (int(ctx.shortlist[idx].sum()), len(idx), 6)

    fig = build_figure(ctx, keep, edges, detail="Outlines", cluster_stats=stats)
    texts = [an.text for an in fig.layout.annotations]
    assert texts, "no cluster labels drawn"
    assert any("shortlisted" in t for t in texts), (
        "labels do not report short-list counts")
    assert any("▲6" in t for t in texts), "labels do not report the change"

    # without stats the label falls back to a plain count, never a fake delta
    plain = build_figure(ctx, keep, edges, detail="Outlines")
    assert all("▲" not in an.text and "▼" not in an.text
               for an in plain.layout.annotations)


def test_dark_is_the_default_theme(env):
    ctx = env["ctx"]
    keep, edges = subgraph("Short list", None, ctx)
    fig = build_figure(ctx, keep, edges, detail="Outlines")
    assert fig.layout.paper_bgcolor == cfg.DARK["paper"]
    assert fig.layout.plot_bgcolor == cfg.DARK["plot"]
    assert cfg.DEFAULT_THEME == "dark"

    light = build_figure(ctx, keep, edges, detail="Outlines", theme="light")
    assert light.layout.paper_bgcolor == cfg.LIGHT["paper"]
    assert light.layout.paper_bgcolor != fig.layout.paper_bgcolor


def test_both_themes_keep_shapes_distinct_and_populations_apart(env):
    """NFR-3: shape carries node type independently of colour, in either theme."""
    assert len(set(cfg.NODE_SYMBOLS.values())) == len(cfg.NODE_SYMBOLS)
    for name, pal in cfg.THEMES.items():
        assert pal["node"]["Applicant"] != pal["node"]["Alum"], name
        assert pal["node"]["Applicant"] != pal["node_dim"], name
        assert len(set(pal["edge"].values())) == 3, name


def test_cluster_fill_still_uniform_in_dark(env):
    """FR-8 survives the theme change: the slider effect is reported in the
    label, never by brightening one disc over another."""
    ctx = env["ctx"]
    keep, edges = subgraph("Short list", None, ctx)
    for theme in ("dark", "light"):
        fig = build_figure(ctx, keep, edges, detail="Outlines", theme=theme)
        fills = {s.fillcolor for s in fig.layout.shapes}
        assert len(fills) == 1, f"{theme}: cluster fills vary, implying rank"


# ------------------------------------- applicant focus and the unfocus path
def _match(env, applicant, limit=5):
    ds, ctx = env["ds"], env["ctx"]
    i = ds.app_index[applicant]
    pairs = []
    for e in ctx.topk_idx[i]:
        if e < 0:
            continue
        e = int(e)
        pairs.append((ds.alum_ids[e], float(ctx.sim[i, e])))
        if len(pairs) >= limit:
            break
    return {"applicant": applicant, "alumni": pairs}


def test_selecting_an_applicant_makes_the_match_unmistakable(env):
    """Layout is frozen, so matched alumni cannot be moved closer. Clarity has
    to come from emphasis plus suppressing everything that is not the answer."""
    ds, ctx = env["ds"], env["ctx"]
    app_id = ds.app_ids[int(np.flatnonzero(ctx.shortlist)[0])]
    mf = _match(env, app_id)
    assert mf["alumni"], "fixture applicant has no matches"

    keep, edges = subgraph("Cohort overview", None, ctx)
    keep = keep | {app_id} | {a for a, _ in mf["alumni"]}
    fig = build_figure(ctx, keep, edges, detail="Outlines", match_focus=mf)

    names = [t.name for t in fig.data]
    assert "Selected applicant" in names
    assert "Similar past students" in names, (
        "matched alumni have no dedicated layer, so they are indistinguishable "
        "from the 600 other squares")

    sel = next(t for t in fig.data if t.name == "Selected applicant")
    assert list(sel.customdata) == [app_id]
    assert sel.marker.line.color == cfg.FOCUS_APPLICANT_RING
    assert sel.marker.size >= cfg.FOCUS_NODE_SIZE

    alum = next(t for t in fig.data if t.name == "Similar past students")
    assert set(alum.customdata) == {a for a, _ in mf["alumni"]}
    assert alum.marker.line.color == cfg.FOCUS_ALUM_RING


def test_focus_suppresses_the_rest_of_the_graph(env):
    ctx, ds = env["ctx"], env["ds"]
    app_id = ds.app_ids[0]
    mf = _match(env, app_id)
    keep, edges = subgraph("Cohort overview", None, ctx)
    keep = keep | {app_id} | {a for a, _ in mf["alumni"]}

    plain = build_figure(ctx, keep, edges, detail="Outlines")
    focused = build_figure(ctx, keep, edges, detail="Outlines", match_focus=mf)

    answer = {"Selected applicant", "Similar past students", "Match"}
    backdrop = [t for t in focused.data if t.name not in answer]
    assert backdrop, "nothing left as context"
    for t in backdrop:
        assert (t.opacity or 1.0) <= cfg.FOCUS_BACKDROP_OPACITY + 1e-9, (
            f"{t.name} is still competing with the answer")
        assert not t.showlegend, f"{t.name} still claims a legend slot"

    # and without focus the same traces are fully visible
    assert max((t.opacity or 1.0) for t in plain.data) > cfg.FOCUS_BACKDROP_OPACITY


def test_match_connectors_scale_with_similarity(env):
    ctx, ds = env["ctx"], env["ds"]
    app_id = ds.app_ids[0]
    mf = _match(env, app_id)
    if len(mf["alumni"]) < 2:
        pytest.skip("need at least two matches to compare widths")
    keep = {app_id} | {a for a, _ in mf["alumni"]}
    fig = build_figure(ctx, keep, [], detail="Connections", match_focus=mf)

    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == len(mf["alumni"])
    widths = [t.line.width for t in lines]
    sims = [s for _, s in mf["alumni"]]
    # ranked descending by similarity, so widths must be non-increasing
    assert sims == sorted(sims, reverse=True)
    assert widths == sorted(widths, reverse=True), (
        f"connector width does not track similarity: {widths}")
    for t in lines:
        assert t.line.color == cfg.FOCUS_EDGE


def test_focus_dims_cluster_labels_without_removing_them(env):
    ctx, ds = env["ctx"], env["ds"]
    mf = _match(env, ds.app_ids[0])
    keep, edges = subgraph("Cohort overview", None, ctx)
    fig = build_figure(ctx, keep, edges, detail="Outlines", match_focus=mf)
    assert fig.layout.annotations, "cluster labels vanished under focus"
    assert all(a.opacity <= 0.4 for a in fig.layout.annotations)


def test_clear_selection_defers_rather_than_writing_widget_keys():
    """Unfocus has to be a request, not a direct write.

    `focus_mode` is bound to a sidebar selectbox, and Streamlit raises
    StreamlitAPIException on any write to a widget's session key after that
    widget has been instantiated. The sidebar is built before the tabs, so a
    button inside a tab cannot reset it directly. `clear_selection` therefore
    only raises a flag, which `app.apply_pending_unfocus` consumes at the top of
    the next run. The consume half is covered end to end by
    tests/test_app.py::test_back_button_returns_to_the_cluster_view.
    """
    import streamlit as st

    from ui.panels import clear_selection

    st.session_state["last_clicked"] = "A26-100000"
    st.session_state["focus_mode"] = "Applicant focus"
    st.session_state["pending_unfocus"] = False

    clear_selection()

    assert st.session_state["pending_unfocus"] is True
    # and it must NOT have touched the widget-bound key itself
    assert st.session_state["focus_mode"] == "Applicant focus"
    assert st.session_state["last_clicked"] == "A26-100000"


def test_match_focus_keeps_alumni_in_view_regardless_of_mode(env):
    """A matched alum must be drawn even when the active focus mode would have
    excluded it, or the connector points at nothing."""
    ctx, ds = env["ctx"], env["ds"]
    app_id = ds.app_ids[0]
    mf = _match(env, app_id)
    keep, edges = subgraph("Clubs at succession risk", None, ctx)
    assert not ({a for a, _ in mf["alumni"]} <= keep), "fixture is not selective"
    keep = keep | {app_id} | {a for a, _ in mf["alumni"]}
    fig = build_figure(ctx, keep, edges, detail="Outlines", match_focus=mf)
    alum = next(t for t in fig.data if t.name == "Similar past students")
    assert len(alum.customdata) == len(mf["alumni"])


# ------------------------------- top-three edges and uniform visual degree
def test_every_person_shows_at_most_three_edges(env):
    """Drawing every edge gave some nodes a visual degree of eight and others
    one, which reads as noise. A uniform degree is what makes the graph look
    homogeneous."""
    from core.graphbuild import important_edges
    ds, ctx = env["ds"], env["ctx"]
    edges = important_edges(ctx)

    per_person: dict[str, int] = {}
    for e in edges:
        for n in (e.source, e.target):
            if n in ds.app_index or n in ds.alum_index:
                per_person[n] = per_person.get(n, 0) + 1

    assert per_person, "no person-attached edges"
    assert max(per_person.values()) <= cfg.TOP_EDGES_PER_PERSON, (
        f"a person shows {max(per_person.values())} edges, above the cap")
    # and it is a genuine reduction
    assert len(edges) < len(ctx.edges) / 2


def test_top_three_mixes_edge_types(env):
    """The three may be any combination of feature types, not a quota per type."""
    from core.graphbuild import important_edges
    edges = important_edges(env["ctx"])
    kinds = {e.kind for e in edges}
    assert len(kinds) >= 3, f"only {kinds} survived, so importance is degenerate"

    ds = env["ds"]
    mixes = 0
    by_person: dict[str, set[str]] = {}
    for e in edges:
        for n in (e.source, e.target):
            if n in ds.app_index or n in ds.alum_index:
                by_person.setdefault(n, set()).add(e.kind)
    mixes = sum(1 for v in by_person.values() if len(v) > 1)
    assert mixes > 100, (
        f"only {mixes} people show more than one edge type; the selection is "
        f"not combining features")


def test_importance_prefers_rare_attachments(env):
    """Rarity is the point: a shared rare activity says more than a shared major
    half the cluster holds."""
    from core.graphbuild import edge_importance, important_edges
    ctx, ds = env["ctx"], env["ds"]

    studies = [e for e in ctx.edges if e.kind == "STUDIES"]
    sports = [e for e in ctx.edges if e.kind == "PLAYS_FOR"]
    assert studies and sports
    mean_studies = np.mean([edge_importance(ctx, e) for e in studies[:200]])
    mean_sport = np.mean([edge_importance(ctx, e) for e in sports[:200]])
    assert mean_sport > mean_studies, (
        "a varsity team is rarer than a major but scores lower")

    # a rare activity must survive selection more often than a hub one
    kept = {(e.source, e.target) for e in important_edges(ctx)}
    hub = min(ds.act_idf, key=ds.act_idf.get)
    rare = max((a for a in ds.act_idf if not a.startswith("ACT-3")),
               key=ds.act_idf.get)
    hub_kept = sum(1 for s, t in kept if t == hub)
    rare_kept = sum(1 for s, t in kept if t == rare)
    hub_total = sum(1 for e in ctx.edges if e.target == hub)
    rare_total = sum(1 for e in ctx.edges if e.target == rare)
    if hub_total and rare_total:
        assert (rare_kept / rare_total) > (hub_kept / hub_total), (
            "hub attachments survive as often as rare ones")


def test_important_edges_are_deterministic(env):
    from core.graphbuild import important_edges
    a = important_edges(env["ctx"])
    b = important_edges(env["ctx"])
    assert [(e.source, e.target, e.kind) for e in a] == \
           [(e.source, e.target, e.kind) for e in b]


# --------------------------------------------------- alternative groupings
def test_major_and_school_group_into_discs(env):
    """The requested change: a major gets its own disc, exactly as a cohort does."""
    ctx = env["ctx"]
    assert set(ctx.assignments) == set(cfg.GROUPINGS)
    try:
        for grouping in cfg.GROUPINGS:
            ctx.apply_grouping(grouping)
            a, geo = ctx.assignment, ctx.cluster_geo
            assert a.members and geo
            assert set(geo) <= set(a.members)
            for g in geo.values():
                assert g["radius"] >= cfg.CLUSTER_DISC_MIN
            # discs must not overlap at any grouping, including the dense one
            items = list(geo.items())
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    (_, ga), (_, gb) = items[i], items[j]
                    gap = math.dist(ga["centre"], gb["centre"]) - (
                        ga["radius"] + gb["radius"])
                    assert gap > 0, f"{grouping}: discs overlap"
    finally:
        ctx.apply_grouping("Cohort")


def test_ring_grows_with_disc_count(env):
    """Grouping by major produces far more discs, so a fixed ring would collide."""
    ctx = env["ctx"]
    try:
        rings = {}
        for grouping in cfg.GROUPINGS:
            ctx.apply_grouping(grouping)
            rings[grouping] = next(iter(ctx.cluster_geo.values()))["ring"]
            counts = len(ctx.cluster_geo)
            assert rings[grouping] >= cfg.CLUSTER_RING_RADIUS
            if counts > 14:
                assert rings[grouping] > cfg.CLUSTER_RING_RADIUS, (
                    f"{grouping} has {counts} discs on the minimum ring")
    finally:
        ctx.apply_grouping("Cohort")


def test_characteristic_edges_never_reach_the_default_view(env):
    """Major, club and athletic-team endpoints are characteristics now, so their
    edges are not drawn at any grouping. The earlier per-grouping redundancy rule
    is subsumed by this and was removed."""
    from core.graphbuild import CHARACTERISTIC_EDGES
    ctx = env["ctx"]
    try:
        for grouping in cfg.GROUPINGS:
            ctx.apply_grouping(grouping)
            keep, edges = subgraph("Cohort overview", None, ctx)
            kinds = {e.kind for e in edges}
            assert not (kinds & CHARACTERISTIC_EDGES), (
                f"{grouping} draws {kinds & CHARACTERISTIC_EDGES}")
            assert kinds == {"SHARES_WITH"}, kinds
    finally:
        ctx.apply_grouping("Cohort")


def test_switching_grouping_is_free_and_reversible(env):
    """NFR-6: every grouping is precomputed, so switching cannot recompute."""
    ctx = env["ctx"]
    before = dict(ctx.layout)
    ctx.apply_grouping("Major")
    assert ctx.layout != before
    assert ctx.grouping == "Major"
    ctx.apply_grouping("Cohort")
    assert ctx.layout == before
    assert ctx.grouping == "Cohort"


def test_every_grouping_is_disclosable(env):
    """XR-12 applies to any grouping: a disc label is an assertion about people."""
    from core import cohorts as CO
    ctx = env["ctx"]
    try:
        for grouping in cfg.GROUPINGS:
            ctx.apply_grouping(grouping)
            defs = CO.grouping_definitions(grouping, ctx.assignment)
            names = {d.name for d in defs}
            for key in ctx.assignment.members:
                assert key in names, f"{grouping}: {key} has no definition"
            for d in defs:
                assert len(d.description) > 15
    finally:
        ctx.apply_grouping("Cohort")


def test_populations_distinguishable_by_size_not_only_shape():
    """Shape alone was not carrying the distinction among 1,800 nodes."""
    assert cfg.NODE_SYMBOLS["Applicant"] != cfg.NODE_SYMBOLS["Alum"]
    big, small = cfg.NODE_SIZES["Alum"], cfg.NODE_SIZES["Applicant"]
    assert big >= small * 1.6, (
        f"alumni at {big}px against applicants at {small}px is too close to read")
    for name, pal in cfg.THEMES.items():
        a, b = pal["node"]["Applicant"], pal["node"]["Alum"]
        assert a != b, name
        # and neither may collide with the rings that mean something else
        assert a not in (cfg.SHORTLIST_OUTLINE, cfg.FOCUS_ALUM_RING), name
        assert b not in (cfg.SHORTLIST_OUTLINE, cfg.FOCUS_ALUM_RING), name


# ------------------- major, clubs and athletics as characteristics, not nodes
def test_default_view_draws_only_people_and_activities(env):
    """The requested change: those three stop competing for the middle of the
    canvas."""
    ctx = env["ctx"]
    keep, _ = subgraph("Cohort overview", None, ctx)
    kinds = {ctx.nodes[n].kind for n in keep}
    assert kinds == cfg.PERSON_KINDS | cfg.CONNECTIVE_KINDS, kinds
    assert not (kinds & cfg.CHARACTERISTIC_KINDS), (
        f"{kinds & cfg.CHARACTERISTIC_KINDS} still drawn as nodes")


@pytest.mark.parametrize("mode,focus", [
    ("Short list", None),
    ("Applicant focus", None),
    ("Cohort isolate", None),
    ("Activity focus", "ACT-01"),
])
def test_person_centric_modes_exclude_characteristic_nodes(env, mode, focus):
    ds, ctx = env["ds"], env["ctx"]
    if mode == "Applicant focus":
        focus = ds.app_ids[int(np.flatnonzero(ctx.shortlist)[0])]
    if mode == "Cohort isolate":
        focus = ctx.assignment.order[0]
    keep, edges = subgraph(mode, focus, ctx)
    drawn = {ctx.nodes[n].kind for n in keep}
    assert not (drawn & cfg.CHARACTERISTIC_KINDS), (
        f"{mode} draws {drawn & cfg.CHARACTERISTIC_KINDS}")
    from core.graphbuild import CHARACTERISTIC_EDGES
    assert not ({e.kind for e in edges} & CHARACTERISTIC_EDGES)


def test_characteristics_live_on_the_person(env):
    """They have to be reachable from the node, since they are no longer nodes."""
    ds, ctx = env["ds"], env["ctx"]
    athletes = 0
    with_clubs = 0
    for pid in ds.app_ids[:400]:
        meta = ctx.nodes[pid].meta
        assert meta["major"], f"{pid} has no major characteristic"
        assert "athlete" in meta and isinstance(meta["athlete"], bool)
        assert isinstance(meta["clubs"], list)
        athletes += meta["athlete"]
        with_clubs += bool(meta["clubs"])
    assert athletes > 0, "no applicant carries athlete status"
    assert with_clubs > 0, "no applicant carries club membership"

    for aid in ds.alum_ids[:200]:
        meta = ctx.nodes[aid].meta
        assert meta["major"]
        assert "seasons" in meta and "captain" in meta


def test_athlete_status_matches_the_source_data(env):
    ds, ctx = env["ds"], env["ctx"]
    for r in ds.applicants.head(300).itertuples(index=False):
        sport = str(r.athletic_recruit_sport or "").strip().lower()
        expected = sport not in ("", "nan", "none")
        assert ctx.nodes[r.applicant_id].meta["athlete"] == expected, r.applicant_id


def test_proxy_sensitive_clubs_are_not_shown_as_keywords(env):
    """§0.1 again: displaying an affinity or religious club as a badge on a person
    is the same inference the scoring exclusion exists to prevent."""
    ds, ctx = env["ds"], env["ctx"]
    banned = {ds.club_name(c) for c in ds.proxy_clubs}
    assert banned
    leaked = [p for p in ds.app_ids
              if set(ctx.nodes[p].meta["clubs"]) & banned]
    assert not leaked, f"{len(leaked)} people show an identity-proxy club badge"

    # and at least one person genuinely holds one, so the filter is load-bearing
    holders = [p for p in ds.app_ids if ds.club_sets.get(p, set()) & ds.proxy_clubs]
    assert holders, "no proxy club membership in the fixture to filter"


def test_clubs_remain_nodes_where_they_are_the_subject(env):
    """AC-22 to AC-24 still need somewhere to hold, so the club views keep them."""
    ctx = env["ctx"]
    ros = ctx.rosters
    cid = ros[ros.at_risk].iloc[0].club_id
    keep, edges = subgraph("Club focus", cid, ctx)
    assert cid in keep
    assert ctx.nodes[cid].kind == "Campus club"
    kinds = {e.kind for e in edges}
    assert kinds & {"MEMBER_OF", "INTERESTED_IN"}, (
        "club view lost the edges that distinguish history from intent")

    keep2, _ = subgraph("Clubs at succession risk", None, ctx)
    assert any(ctx.nodes[n].kind == "Campus club" for n in keep2)


def test_removed_focus_modes_are_gone_from_the_menu():
    """Major cohort and Club focus depended on nodes that no longer render by
    default; leaving them selectable would produce empty views."""
    assert "Major cohort" not in cfg.FOCUS_MODES
    assert "Club focus" not in cfg.FOCUS_MODES
    assert "Clubs at succession risk" in cfg.FOCUS_MODES


def test_adjacency_still_carries_characteristics(env):
    """They stop being drawn, not stop existing: XR-6 shared-path highlighting
    and the club panels both read them from adjacency."""
    from core.graphbuild import shared_nodes
    ds, ctx = env["ds"], env["ctx"]
    a = ds.app_ids[0]
    e = ds.alum_ids[int(ctx.topk_idx[0, 0])]
    shared = shared_nodes(ctx, a, e)
    kinds = {ctx.nodes[s].kind for s in shared}
    assert shared, "shared-path highlighting lost its evidence"
    # majors and teams may legitimately appear as shared evidence
    assert kinds - cfg.CONNECTIVE_KINDS or kinds & cfg.CONNECTIVE_KINDS


def test_badge_helper_renders_every_characteristic():
    from ui.panels import _badge
    for kind in cfg.BADGE:
        html = _badge("x", kind)
        bg, fg = cfg.BADGE[kind]
        assert bg in html and fg in html
    assert cfg.ATHLETE_LABEL != cfg.NON_ATHLETE_LABEL
