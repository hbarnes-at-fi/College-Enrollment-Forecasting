"""Tab bodies. Streamlit lives here and in app.py, never in core/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg                                              # noqa: E402
from core import clubs as club_mod                                # noqa: E402
from core import explain, priorities, similarity                  # noqa: E402
from core.graphbuild import shared_nodes, subgraph                # noqa: E402
from core import cohorts as cohort_mod                             # noqa: E402
from ui.graphview import (build_figure, detail_caption,            # noqa: E402
                          legend_caption, selected_ids)

MONEY = "${:,.0f}"


def clear_selection() -> None:
    """Request unfocus.

    `focus_mode` is bound to a sidebar widget, and Streamlit refuses writes to a
    widget's session key once that widget has been instantiated this run. The
    sidebar is built before the tabs, so the reset has to be deferred: this sets
    a flag that `app.apply_pending_unfocus()` consumes at the top of the next
    run, before any widget exists.
    """
    st.session_state["pending_unfocus"] = True


def match_focus_for(c, applicant_id: str, limit: int = 5) -> dict:
    """The payload that drives the emphasis layer: who this applicant matches."""
    ds = c.ds
    i = ds.app_index[applicant_id]
    pairs = []
    for e in c.topk_idx[i]:
        if e < 0:
            continue
        e = int(e)
        pairs.append((ds.alum_ids[e], float(c.sim[i, e])))
        if len(pairs) >= limit:
            break
    return {"applicant": applicant_id, "alumni": pairs}


# ------------------------------------------------------------------ helpers
def _badge(text: str, kind: str) -> str:
    bg, fg = cfg.BADGE.get(kind, cfg.BADGE["student"])
    return (f"<span style='background:{bg};color:{fg};padding:3px 9px;"
            f"border-radius:11px;font-size:0.80rem;font-weight:600;"
            f"white-space:nowrap;display:inline-block;margin:0 5px 5px 0'>"
            f"{text}</span>")


def characteristic_badges(c, nid: str) -> None:
    """Major, athletic status and clubs as highlighted keywords.

    These were four node types crowding the middle of the canvas and
    contributing edges that said little. As keywords on the selected person they
    read immediately and cost the graph nothing.
    """
    meta = c.ctx.nodes[nid].meta
    chips: list[str] = []

    if meta.get("cohort"):
        chips.append(_badge(meta["cohort"], "cohort"))
    if meta.get("major"):
        chips.append(_badge(meta["major"], "major"))

    if meta.get("athlete"):
        sport = meta.get("sport") or "varsity"
        extra = ""
        if meta.get("captain"):
            extra = ", captain"
        elif meta.get("seasons"):
            extra = f", {meta['seasons']} seasons"
        chips.append(_badge(f"{cfg.ATHLETE_LABEL} · {sport}{extra}", "athlete"))
    else:
        chips.append(_badge(cfg.NON_ATHLETE_LABEL, "student"))

    # rarest first, so the informative activities lead
    for activity in meta.get("activities", [])[:6]:
        chips.append(_badge(activity, "activity"))
    for club in meta.get("clubs", [])[:6]:
        chips.append(_badge(club, "club"))

    st.markdown("".join(chips), unsafe_allow_html=True)
    if not meta.get("clubs"):
        st.caption("No campus club interest on record.")


def _bar(label: str, value: float, total: float, suffix: str = "") -> None:
    """Proportional bar. XR-3 keeps raw coefficients out of the default view."""
    frac = 0.0 if total <= 0 else max(0.0, min(1.0, value / total))
    st.write(f"{label} {suffix}")
    st.progress(frac)


def _neighbour_rows(c, a_idx: int, limit: int = 5) -> list[dict]:
    ds = c.ds
    out = []
    for e in c.topk_idx[a_idx]:
        if e < 0:
            continue
        e = int(e)
        dec = similarity.decompose(c.M, c.A, c.w, a_idx, e)
        pct = float((c.sim[a_idx] <= c.sim[a_idx, e]).mean())
        det = explain.shared_detail(ds, ds.app_ids[a_idx], ds.alum_ids[e])
        alum = ds.alumni.iloc[e]
        out.append({
            "alum_id": ds.alum_ids[e],
            "similarity": float(c.sim[a_idx, e]),
            "outcome": str(alum.success_composite_tier),
            "final_gpa": float(alum.final_cum_gpa),
            "exemplar": alum.exemplar_flag == "Y",
            "why": explain.rationale(dec, pct, c.w, det),
            "dec": dec,
        })
        if len(out) >= limit:
            break
    return out


# -------------------------------------------------------------- graph tab
def graph_tab(c) -> None:
    left, right = st.columns([7, 3], gap="medium")

    selected = st.session_state.get("last_clicked")
    focusing = selected in c.ds.app_index

    with left:
        if selected or st.session_state.get("selected_pair"):
            back = st.columns([1.4, 4])
            with back[0]:
                if st.button("← Back to clusters", width="stretch",
                             type="primary"):
                    clear_selection()
                    st.rerun()
            with back[1]:
                if focusing:
                    st.caption(f"Showing **{selected}** and the past students "
                               f"most like them. Everything else is dimmed.")

        focus = None
        mode = st.session_state["focus_mode"]
        ds = c.ds
        if mode == "Cohort isolate":
            groups = [x for x in c.ctx.assignment.order
                      if x in c.ctx.assignment.members]
            focus = st.selectbox(c.ctx.grouping, groups, key="focus_cohort")
        elif mode == "Applicant focus":
            pool = [ds.app_ids[i] for i in np.flatnonzero(c.mask)] or ds.app_ids
            focus = st.selectbox("Applicant", pool, key="focus_applicant")
        elif mode == "Activity focus":
            opts = {n.label: n.id for n in c.ctx.nodes.values()
                    if n.kind in ("Activity", "Athletic team")}
            focus = opts[st.selectbox("Activity", sorted(opts),
                                      key="focus_activity")]
        elif mode == "Major cohort":
            opts = sorted({n.label for n in c.ctx.nodes.values()
                           if n.kind == "Major"})
            focus = "MAJ::" + st.selectbox("Major", opts, key="focus_major")
        elif mode == "Club focus":
            opts = {n.label: n.id for n in c.ctx.nodes.values()
                    if n.kind == "Campus club"}
            focus = opts[st.selectbox("Club", sorted(opts), key="focus_club")]

        if mode == "Full population":
            st.warning(
                "Full population adds every inferred similarity edge on top of "
                "the observed ones. Dense by design; Cohort overview shows the "
                "same people without the inference layer (NR-3c, NR-5c).")

        keep, edges = subgraph(mode, focus, c.ctx)
        if not st.session_state["show_clubs"]:
            keep = {n for n in keep if c.ctx.nodes[n].kind != "Campus club"}
            edges = [e for e in edges if e.source in keep and e.target in keep]

        highlight = emphasis = None
        pair = st.session_state.get("selected_pair")
        if pair and pair[0] in keep and pair[1] in keep:
            sh = shared_nodes(c.ctx, pair[0], pair[1])
            highlight = sh | set(pair)
            emphasis = ({(pair[0], s) for s in sh}
                        | {(pair[1], s) for s in sh} | {tuple(pair)})

        detail = st.session_state["detail_level"]
        focused = bool(pair) or focusing or mode in ("Applicant focus",
                                                     "Cohort isolate")
        mf = match_focus_for(c, selected) if focusing else None
        if mf:
            # keep the matched alumni in view even if the current mode would
            # have excluded them
            keep = keep | {selected} | {a for a, _ in mf["alumni"]}

        fig = build_figure(
            c.ctx, keep, edges, highlight, emphasis,
            height=700,
            detail=detail,
            show_labels=st.session_state["show_cohort_labels"],
            isolate=focus if mode == "Cohort isolate" else None,
            focused=focused,
            theme="dark" if st.session_state["dark_graph"] else "light",
            cluster_stats=None if mf else c.cluster_stats,
            match_focus=mf)
        event = st.plotly_chart(fig, width="stretch", key="graph",
                                on_select="rerun", selection_mode=("points",))
        drawn = len(fig.data) and sum(
            1 for t in fig.data if getattr(t, "mode", "") == "lines")
        st.caption(f"{len(keep):,} nodes. {detail_caption(detail, c.ctx.clustered, c.ctx.grouping)}")
        if not drawn:
            st.caption("No connections are being drawn at this detail level. "
                       "They exist; raise Detail to see them.")
        st.caption(legend_caption())

        picked = selected_ids(event)
        if picked and picked[0] != selected:
            st.session_state["last_clicked"] = picked[0]
            st.session_state["selected_pair"] = None
            st.rerun()

    with right:
        nid = st.session_state.get("last_clicked")
        if not nid or nid not in c.ctx.nodes:
            st.info("Click any node to inspect it. Click an applicant to see "
                    "which past students they most resemble, and how those "
                    "students actually did here.")
            return
        if st.button("← Back to clusters", key="back_panel", width="stretch"):
            clear_selection()
            st.rerun()
        node = c.ctx.nodes[nid]
        kind_label = {"Applicant": "Applicant",
                      "Alum": "Past student"}.get(node.kind, node.kind)
        st.markdown(f"**{node.label}** · {kind_label}")

        if node.kind in cfg.PERSON_KINDS:
            characteristic_badges(c, nid)
            # NR-4d: position is single-valued, membership is not
            others = [x for x in (node.meta.get("cohorts") or [])
                      if x != node.meta.get("cohort")]
            if others:
                st.caption(f"Also matches {', '.join(others)}")

        if node.kind == "Applicant":
            _applicant_panel(c, nid)
        elif node.kind == "Alum":
            _alum_panel(c, nid)
        elif node.kind == "Campus club":
            club_detail(c, nid)
        else:
            deg = len(c.ctx.adjacency.get(nid, set()))
            st.write(f"Connected to {deg:,} people in the current data.")


def _applicant_panel(c, nid: str) -> None:
    ds = c.ds
    a_idx = ds.app_index[nid]
    on = bool(c.mask[a_idx])
    st.markdown("On the short list" if on else "Not on the short list")

    row = ds.applicants.iloc[a_idx]
    st.caption(f"{row.intended_major} · {row.region} · reader rating "
               f"{row.academic_rating} · {row.family_income_band}")

    st.markdown(explain.pick_rationale(
        ds, a_idx, c.priority_pct(a_idx), c.priority_weights,
        c.carried.get(nid), bool(c.thin[a_idx]), in_list=on))

    cf = st.session_state.get("counterfactuals", {}).get(nid)
    if cf:
        st.markdown(f"_{cf}_")

    rows = _neighbour_rows(c, a_idx)
    st.markdown("### Past students most like them")
    if not rows:
        st.warning("No alumni clear the current match strictness. Lower it, or "
                   "widen the similarity dimensions, to get a comparison set.")
        return
    if bool(c.thin[a_idx]):
        st.warning(cfg.THIN_NEIGHBOURHOOD_NOTE)

    st.caption(f"Ringed in green on the graph, connected to {nid} by a bright "
               f"line. Thicker line means closer match.")

    for rank, r in enumerate(rows):
        label = (cfg.FOCUS_RANK_LABELS[rank]
                 if rank < len(cfg.FOCUS_RANK_LABELS) else f"#{rank + 1}")
        alum = ds.alumni.iloc[ds.alum_index[r["alum_id"]]]
        with st.container(border=True):
            head = st.columns([3, 2])
            with head[0]:
                st.markdown(f"**{r['alum_id']}**")
                st.caption(f"{label} match · {alum.final_major} · "
                           f"class of {alum.cohort_year}")
            with head[1]:
                st.markdown(f"**{r['outcome'].split(' - ')[-1]}**"
                            + ("  ·  exemplar" if r["exemplar"] else ""))
                st.caption(f"Final GPA {r['final_gpa']:.2f} · growth "
                           f"{float(alum.gpa_delta):+.2f}")
            st.progress(min(1.0, r["similarity"]),
                        text=f"similarity {r['similarity']:.0%}")

            shared = shared_nodes(c.ctx, nid, r["alum_id"])
            if shared:
                names = [c.ctx.nodes[s].label for s in sorted(shared)]
                st.caption("Shared: " + " · ".join(names[:6]))
            st.caption(r["why"])
            if st.button("Break this match down", key=f"pair_{nid}_{r['alum_id']}",
                         width="stretch"):
                st.session_state["selected_pair"] = (nid, r["alum_id"])
                st.rerun()

    tiers = [r["outcome"] for r in rows]
    good = sum(1 for t in tiers if "Thrived" in t or "Strong" in t)
    st.info(f"{good} of {len(rows)} closest past students landed in the top two "
            f"outcome tiers. That is the evidence behind this applicant's "
            f"academic strength score, and it is the whole argument for "
            f"comparing against outcomes rather than credentials.")


def _alum_panel(c, nid: str) -> None:
    ds = c.ds
    e_idx = ds.alum_index[nid]
    row = ds.alumni.iloc[e_idx]
    st.caption(f"{row.final_major} · class of {row.cohort_year} · "
               f"{row.success_composite_tier}")
    cols = st.columns(2)
    cols[0].metric("Final GPA", f"{float(row.final_cum_gpa):.2f}")
    cols[1].metric("Growth", f"{float(row.gpa_delta):+.2f}")
    st.caption(f"Retained: {row.retained_year2} · graduated in four: "
               f"{row.graduated_4yr} · leadership roles: "
               f"{row.leadership_roles_count}")
    st.caption("Applicants are compared against this alum's ENTRY profile only "
               "(DR-4a). Their outcome is the label, never a comparison field.")


# ----------------------------------------------------------- why / XR-1
def why_tab(c) -> None:
    st.markdown("#### Where a score comes from")
    pair = st.session_state.get("selected_pair")
    ds = c.ds
    if not pair:
        st.info("Pick an applicant in the Graph tab, then click one of its "
                "closest alumni, to decompose that match here.")
        return

    a_id, e_id = pair
    a_idx, e_idx = ds.app_index[a_id], ds.alum_index[e_id]
    dec = similarity.decompose(c.M, c.A, c.w, a_idx, e_idx)
    pct = float((c.sim[a_idx] <= c.sim[a_idx, e_idx]).mean())
    det = explain.shared_detail(ds, a_id, e_id)

    st.markdown(f"**{a_id} → {e_id}**")
    st.markdown(explain.rationale(dec, pct, c.w, det))

    total = dec["score"]
    st.markdown("**Contributions** (these sum to the score, XR-1)")
    for name, val in sorted(dec["contributions"].items(), key=lambda kv: -kv[1]):
        _bar(name, val, total, f"— {val / total:.0%} of the match"
             if total > 0 else "")
    if dec["dropped"]:
        st.caption("Not compared for this pair, so excluded from the "
                   f"normaliser (SR-7b): {', '.join(dec['dropped'])}")

    if st.session_state["disclose_weights"]:
        st.markdown("**Numeric detail** (disclosed on request, XR-3)")
        st.dataframe(pd.DataFrame([
            {"dimension": n, "match": dec["matches"][n],
             "weight": float(c.w[cfg.DIM_NAMES.index(n)]),
             "contribution": v}
            for n, v in dec["contributions"].items()]),
            hide_index=True, width="stretch")

    sh = shared_nodes(c.ctx, a_id, e_id)
    st.markdown("**Shared evidence** (XR-6)")
    if sh:
        st.write(", ".join(sorted(c.ctx.nodes[s].label for s in sh)))
        if st.button("Highlight this in the graph"):
            st.session_state["selected_pair"] = (a_id, e_id)
            st.session_state["focus_mode"] = "Applicant focus"
            st.session_state["focus_applicant"] = a_id
            st.rerun()
    else:
        st.write("No observed nodes in common. The similarity is entirely "
                 "attribute-based, which is weaker evidence.")


# ----------------------------------------------------- short list / XR-10
def shortlist_tab(c) -> None:
    ds = c.ds
    idx = np.flatnonzero(c.mask)
    st.markdown(f"#### {len(idx):,} applicants in the discussion set")
    st.caption("Advisory only. This is a discussion set, not a decision, and it "
               "carries no recommendation to admit or deny (FR-4).")

    rows = []
    for i in idx:
        r = ds.applicants.iloc[i]
        rows.append({
            "applicant_id": ds.app_ids[i],
            "school": r.intended_school,
            "major": r.intended_major,
            "region": r.region,
            "rating": int(r.academic_rating),
            "yield": float(r.predicted_yield_prob),
            "net price": float(r.net_price),
            "first gen": r.first_generation,
            "income band": r.family_income_band,
            "carries at-risk club": ", ".join(
                ds.club_name(x) for x in c.carried.get(ds.app_ids[i], [])),
            "credentials only": bool(c.thin[i]),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch", height=420)

    st.download_button(
        "Export the discussion set with its configuration",
        df.assign(**{"_configuration": json.dumps(c.config_blob())}
                  ).to_csv(index=False),
        file_name="discussion_set.csv", mime="text/csv")

    pick = st.selectbox("Export a single applicant's reasoning",
                        [""] + df.applicant_id.tolist())
    if pick:
        a_idx = ds.app_index[pick]
        nb = _neighbour_rows(c, a_idx)
        payload = explain.export_payload(
            ds, a_idx, c.priority_words, c.dim_words,
            [(r["alum_id"], r["similarity"], r["why"]) for r in nb],
            bool(c.mask[a_idx]),
            explain.pick_rationale(ds, a_idx, c.priority_pct(a_idx),
                                   c.priority_weights, c.carried.get(pick),
                                   bool(c.thin[a_idx]), bool(c.mask[a_idx])),
            c.cost)
        st.json(payload, expanded=False)
        st.download_button(f"Download {pick}.json",
                           json.dumps(payload, indent=2),
                           file_name=f"{pick}.json", mime="application/json")


# ---------------------------------------------------------- clubs / IR-7
def club_detail(c, cid: str) -> None:
    ds = c.ds
    row = c.ros[c.ros.club_id == cid].iloc[0]
    st.caption(f"{row.category} · charter {row.charter_status} · "
               f"advisor backed {row.advisor_backed}")
    if row.proxy_sensitive:
        st.warning("Affinity or religious organisation. Projected and shown "
                   "here, but excluded from similarity scoring and from "
                   "diversity coverage credit: interest in it is a proxy for "
                   "race or religion (specs §0.1).")
    cols = st.columns(3)
    cols[0].metric("Roster outlook", row.band)
    cols[1].metric("Members staying", int(row.holdover))
    cols[2].metric("Needs at least", int(row.min_viable_members))
    st.caption("Roster outlook is probability-weighted, so it is an "
               "expectation and not a headcount (SR-8d).")

    interested = club_mod.interested_applicants(ds, cid)
    members = club_mod.member_alumni(ds, cid)
    st.markdown(f"**{len(interested)} interested applicants · "
                f"{len(members)} alumni members**")
    if interested:
        sub = ds.applicants[ds.applicants.applicant_id.isin(interested)]
        st.dataframe(
            pd.DataFrame({
                "applicant_id": sub.applicant_id,
                "yield": sub.predicted_yield_prob,
                "on short list": [bool(c.mask[ds.app_index[p]])
                                  for p in sub.applicant_id]}),
            hide_index=True, width="stretch", height=180)
    if members:
        sub = ds.alumni[ds.alumni.alum_id.isin(members)]
        st.caption("Alumni who were members, and how they did: "
                   + ", ".join(f"{r.alum_id} ({r.success_composite_tier})"
                               for r in sub.head(6).itertuples(index=False)))


# ------------------------------------ cohort clusters / XR-12, IR-8e
def cohort_panel(c) -> None:
    a = c.ctx.assignment
    if a is None:
        return
    st.markdown(f"#### {c.ctx.grouping} clusters (XR-12)")
    st.caption("A cluster label is an assertion about people, and a reader will "
               "treat spatial grouping as fact. Every rule is stated here, "
               "including the two that are conjunctions rather than single "
               "conditions. Ring position carries no ranking (FR-8).")

    defs = {d.name: d for d in
            cohort_mod.grouping_definitions(c.ctx.grouping, a)}
    rows = []
    for i, name in enumerate(a.order, start=1):
        if name not in a.members:
            continue
        app, alum = a.split(c.ds, name)
        rows.append({
            "precedence": i,
            "group": name,
            "applicants": len(app),
            "alumni": len(alum),
            "on short list": sum(1 for p in app if c.mask[c.ds.app_index[p]]),
            "definition": defs[name].description,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.caption(
        "Precedence is computed by ascending predicate breadth, rarest first "
        "(NR-4g). It is not hand-ordered: a hand-picked order buried the access "
        "cohort behind broader school-track predicates until it fell below the "
        "minimum size and vanished. Rarity ordering keeps the most distinctive "
        "groups large enough to read without anyone deciding which group "
        "matters more.")
    if a.merged:
        st.caption(f"Merged into the general cohort for being too small to read "
                   f"(NR-4e): {', '.join(a.merged)}.")

    st.info("No cohort is defined on ethnicity or sex: those columns are "
            "dropped when the data loads, so a cohort rule cannot reference "
            "them. Affinity and religious club membership is also excluded. "
            "Spatial clustering turns a rule into a picture, and a cluster "
            "built on either would be a segregation map (FR-7).")

    pick = st.selectbox("Inspect a cluster",
                        [""] + [r["group"] for r in rows])
    if pick:
        app, alum = a.split(c.ds, pick)
        st.markdown(f"**{pick}** — {defs[pick].description}")
        cols = st.columns(3)
        cols[0].metric("Applicants", len(app))
        cols[1].metric("Alumni", len(alum))
        cols[2].metric("On short list", sum(
            1 for p in app if c.mask[c.ds.app_index[p]]))

        sub = c.ds.alumni[c.ds.alumni.alum_id.isin(alum)]
        if len(sub):
            st.caption(
                "How this cohort's alumni actually did: mean final GPA "
                f"{np.asarray(sub.final_cum_gpa, dtype=float).mean():.2f}, "
                f"{(np.asarray(sub.exemplar_flag) == 'Y').mean():.0%} flagged "
                "exemplars. This is the comparison clustering exists to enable "
                "(NR-4a).")

        appsub = c.ds.applicants[c.ds.applicants.applicant_id.isin(app)]
        if len(appsub):
            st.markdown("Composition of this cluster against class targets")
            frames = []
            for field in cfg.COMPOSITION_TARGETS:
                share = appsub[field].value_counts(normalize=True)
                for cat, tgt in st.session_state["targets"][field].items():
                    frames.append({"dimension": field, "category": cat,
                                   "in cluster": f"{share.get(cat, 0.0):.0%}",
                                   "class target": f"{tgt:.0%}"})
            st.dataframe(pd.DataFrame(frames), hide_index=True,
                         width="stretch", height=240)


# ------------------------------------------- composition / FR-1, FR-3
def composition_tab(c) -> None:
    ds = c.ds
    st.markdown("#### Class projection against targets (SR-5c)")
    st.caption("Targets are editable here, and the diversity priority scores "
               "against them. A diversity score measured against undisclosed "
               "targets is not explainable, which is why they are on screen "
               "rather than buried in configuration.")

    targets = st.session_state["targets"]
    realised = priorities.composition_of(c.C, c.mask, c.labels)
    edited = False

    for field, spec in targets.items():
        st.markdown(f"**{field.replace('_', ' ').title()}**")
        got = realised.get(field, {})
        frame = pd.DataFrame([
            {"category": k, "target %": round(v * 100, 1),
             "short list %": round(got.get(k, 0.0) * 100, 1),
             "gap": round((got.get(k, 0.0) - v) * 100, 1)}
            for k, v in spec.items()])
        out = st.data_editor(
            frame, hide_index=True, width="stretch",
            key=f"target_editor_{field}",
            disabled=["category", "short list %", "gap"],
            column_config={
                "target %": st.column_config.NumberColumn(
                    "target %", min_value=0.0, max_value=100.0, step=0.5,
                    help="Edit to change what diversity is measured against."),
            })
        new = {r["category"]: float(r["target %"]) / 100.0
               for _, r in out.iterrows()}
        if any(abs(new[k] - spec[k]) > 1e-9 for k in spec):
            # Store the raw edit. Normalising here would write back a value
            # different from what the editor holds, which re-triggers this same
            # branch on the next run and loops forever. Normalisation happens at
            # consumption, inside composition_matrix.
            targets[field] = new
            edited = True

    if edited:
        st.session_state["targets"] = targets

    effective = priorities.normalise_targets(targets)
    total_gap = sum(abs(realised.get(f, {}).get(k, 0.0) - v)
                    for f, spec in effective.items() for k, v in spec.items())
    st.caption(f"Total distance from target composition: {total_gap:.3f}. "
               "Diversity minimises this, one pick at a time. Edited targets "
               "are renormalised per dimension when scored, so they need not "
               "sum to 100 as you type.")
    if st.button("Restore default targets"):
        st.session_state["targets"] = {f: dict(s) for f, s in
                                       cfg.COMPOSITION_TARGETS.items()}
        st.rerun()

    cohort_panel(c)

    st.markdown("#### Exemplar cohort beside the applicant pool (FR-1)")
    st.caption("Similarity to past success is a bias amplifier by construction. "
               "If the exemplar cohort over-represents a school type, region or "
               "income band, the graph promotes applicants who resemble them and "
               "calls it merit. Visible skew is the warning this tool owes you.")

    ex = ds.alumni[ds.alumni.exemplar_flag == "Y"]
    pairs = [("region", "entry_region"), ("hs_type", "entry_hs_type"),
             ("first_generation", "entry_first_generation"),
             ("family_income_band", "entry_family_income_band")]
    for app_col, alum_col in pairs:
        a = ds.applicants[app_col].value_counts(normalize=True)
        b = ex[alum_col].value_counts(normalize=True)
        keys = sorted(set(a.index) | set(b.index))
        frame = pd.DataFrame([{
            "category": k,
            "applicant pool": f"{a.get(k, 0.0):.0%}",
            "exemplar cohort": f"{b.get(k, 0.0):.0%}",
            "skew": f"{b.get(k, 0.0) - a.get(k, 0.0):+.0%}"} for k in keys])
        st.markdown(f"**{app_col.replace('_', ' ').title()}**")
        st.dataframe(frame, hide_index=True, width="stretch")

        worst = max(keys, key=lambda k: abs(b.get(k, 0.0) - a.get(k, 0.0)))
        gap = b.get(worst, 0.0) - a.get(worst, 0.0)
        if abs(gap) >= 0.08:
            st.warning(f"FR-3 drift: '{worst}' is {abs(gap):.0%} "
                       f"{'over' if gap > 0 else 'under'}-represented among "
                       f"exemplars relative to the pool. Academic strength "
                       f"validated against this cohort inherits that skew.")


# ------------------------------------------------- what changed / XR-4
def changed_tab(c) -> None:
    st.markdown("#### What your last change did (XR-4)")
    ch = st.session_state.get("last_change")
    if not ch:
        st.info("Move a priority control and the consequence appears here: who "
                "entered, who left, and which control caused it. With no "
                "numbers on the controls, observing the effect is how you "
                "understand them.")
    else:
        st.caption(f"Controls changed: {', '.join(ch['changed_controls'])}")
        cols = st.columns(2)
        cols[0].metric("Entered", ch["n_entered"])
        cols[1].metric("Left", ch["n_left"])
        both = st.columns(2)
        with both[0]:
            st.markdown("**Entered**")
            st.dataframe(pd.DataFrame(ch["entered"]) if ch["entered"]
                         else pd.DataFrame(columns=["applicant_id", "cause"]),
                         hide_index=True, width="stretch", height=260)
        with both[1]:
            st.markdown("**Left**")
            st.dataframe(pd.DataFrame(ch["left"]) if ch["left"]
                         else pd.DataFrame(columns=["applicant_id", "cause"]),
                         hide_index=True, width="stretch", height=260)

    st.markdown("#### What this configuration costs (XR-5)")
    st.caption("Any single priority pushed to decisive buys its own measure and "
               "sells the other three. A tool that let you max one priority "
               "without showing the cost would be persuasive, not advisory.")
    s, b, d = c.cost["shortlist"], c.cost["baseline"], c.cost["delta"]
    st.dataframe(pd.DataFrame([
        {"measure": "Mean reader rating",
         "short list": f"{s['academic']:.2f}", "whole pool": f"{b['academic']:.2f}",
         "change": f"{d['academic']:+.2f}"},
        {"measure": "Mean probability of enrolling",
         "short list": f"{s['yield']:.3f}", "whole pool": f"{b['yield']:.3f}",
         "change": f"{d['yield']:+.3f}"},
        {"measure": "Mean net price",
         "short list": MONEY.format(s["net_price"]),
         "whole pool": MONEY.format(b["net_price"]),
         "change": MONEY.format(d["net_price"])},
        {"measure": "First-generation share",
         "short list": f"{s['first_gen']:.0%}", "whole pool": f"{b['first_gen']:.0%}",
         "change": f"{d['first_gen']:+.0%}"},
        {"measure": "Low-income share",
         "short list": f"{s['low_income']:.0%}",
         "whole pool": f"{b['low_income']:.0%}",
         "change": f"{d['low_income']:+.0%}"},
    ]), hide_index=True, width="stretch")

    if s["low_income"] <= 0.06:
        st.error("Low-income representation in this short list is near zero. "
                 "That is the measured consequence of prioritising net revenue, "
                 "not an artefact.")


# -------------------------------------------------------- caveats / FR-6
def caveats_panel() -> None:
    """FR-6. Lives in the sidebar now that the tab is gone: the limitations must
    be visible in the app, not only in the requirements document, because a demo
    audience would not otherwise know."""
    st.markdown("### Before you believe anything here")
    for title, body in cfg.MOCK_DATA_CAVEATS:
        with st.expander(title, expanded=False):
            st.write(body)
    with st.expander("The short list is greedy, not optimal"):
        st.write("When diversity carries weight, picks are made one at a time, "
                 "each locally best given prior picks. No individual is "
                 "diverse; an applicant only contributes relative to who is "
                 "already selected, which makes the exact problem "
                 "combinatorial and unsolvable inside an interactive budget. "
                 "A different pick order could produce a slightly better class.")
    with st.expander("Similarity is deliberately simple"):
        st.write("Scores are an additive weighted mean over named dimensions, "
                 "so every score decomposes into reasons that sum to the whole. "
                 "Graph neural networks and learned embeddings would likely "
                 "score better and were excluded anyway, because their output "
                 "cannot be interrogated by a committee.")
    with st.expander("What is excluded from scoring, and why"):
        st.write("Ethnicity and sex are dropped when the data is loaded, so "
                 "they cannot reach a score by later carelessness. Affinity and "
                 "religious clubs are projected and displayed but excluded from "
                 "similarity and from diversity coverage credit, because "
                 "interest in one is a proxy for race or religion. Alumni "
                 "giving is not captured at all: it is a wealth proxy, and "
                 "including it would teach the graph to prefer applicants who "
                 "resemble affluent alumni.")
    with st.expander("Advisory only"):
        st.write("Nothing here is an admission decision. There is no ranked "
                 "admit or deny output and no automated cut. Every export "
                 "carries the configuration that produced it, so no number "
                 "travels without its reasoning.")
