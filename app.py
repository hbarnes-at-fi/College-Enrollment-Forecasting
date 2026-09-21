"""Interactive admissions similarity graph.

Advisory only. Produces a discussion set, never an admission decision (FR-4).

Run: streamlit run app.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg                                             # noqa: E402
from core import explain, priorities, similarity                 # noqa: E402
from core.clubs import (coverage_matrix, project_rosters,         # noqa: E402
                        sole_pipeline_applicants)
from core.features import build_tensors, flatten                 # noqa: E402
from core.graphbuild import build_context                        # noqa: E402
from core.loader import load, suppression_audit                  # noqa: E402
from ui import panels                                            # noqa: E402

st.set_page_config(page_title="Admissions similarity graph",
                   layout="wide", initial_sidebar_state="collapsed")


# --------------------------------------------------------------- bootstrap
@st.cache_resource(show_spinner="Loading data and precomputing similarity…")
def bootstrap():
    """Everything that does not depend on a control value.

    The match tensor must never be keyed on weights: build is cached here,
    contraction runs per interaction and costs about 8 ms.
    """
    ds = load()
    M, A = build_tensors(ds)
    M2, A2 = flatten(M, A)
    ctx = build_context(ds)
    ros = project_rosters(ds)
    B, club_ids = coverage_matrix(ds, ros)
    return SimpleNamespace(
        ds=ds, M=M, A=A, M2=M2, A2=A2, ctx=ctx, ros=ros, B=B,
        club_ids=club_ids,
        id_rank=np.argsort(np.argsort(np.asarray(ds.app_ids))),
        carried=sole_pipeline_applicants(ds, ros),
        audit=suppression_audit(ds),
        p1=priorities.percentile(priorities.p1_yield(ds)),
        p2=priorities.percentile(priorities.p2_expected_ntr(ds)),
    )


boot = bootstrap()
DS = boot.ds


# ------------------------------------------------------------------ state
def init_state() -> None:
    defaults = {
        **cfg.PRESETS["Balanced"],
        "preset": "Balanced",
        "shortlist_size": "Selective",
        "w_club_cov": "Minor",
        "match_strictness": "Moderate",
        "matches_per_applicant": "A few",
        "show_clubs": True,

        "show_sole_pipeline": False,
        "focus_mode": "Cohort overview",
        "detail_level": cfg.DETAIL_DEFAULT,
        "clustered": True,
        "group_by": cfg.GROUP_BY_DEFAULT,
        "show_cohort_labels": True,
        "dark_graph": cfg.DEFAULT_THEME == "dark",
        "isolate_cohort": None,
        "prev_mask": None,
        "pending_unfocus": False,
        "disclose_weights": False,
        "selected_pair": None,
        "last_clicked": None,
        "last_change": None,
        "counterfactuals": {},
        # SR-5c: editable, so they live in state rather than only in config
        "targets": {f: dict(s) for f, s in cfg.COMPOSITION_TARGETS.items()},
    }
    for label in priorities.SUCCESS_COMPONENTS:
        defaults[f"succ_{label}"] = True
    for name, dflt in cfg.DIMENSIONS:
        defaults[f"dim_on_{name}"] = dflt != "Not considered"
        defaults[f"dim_w_{name}"] = dflt if dflt != "Not considered" else "Moderate"
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


def apply_pending_unfocus() -> None:
    """Consume a deferred unfocus request before any widget is instantiated.

    Must run here, not in the button handler: `focus_mode` is a widget key and
    Streamlit rejects writes to it once the sidebar has been built.
    """
    if not st.session_state.get("pending_unfocus"):
        return
    st.session_state["pending_unfocus"] = False
    st.session_state["last_clicked"] = None
    st.session_state["selected_pair"] = None
    st.session_state["counterfactuals"] = {}
    st.session_state["focus_mode"] = "Cohort overview"


apply_pending_unfocus()


def apply_preset() -> None:
    name = st.session_state["preset"]
    for key, word in cfg.PRESETS[name].items():
        st.session_state[key] = word


def reset_defaults() -> None:
    for key in list(st.session_state):
        if key not in ("preset",):
            del st.session_state[key]
    st.session_state["preset"] = "Balanced"
    init_state()


# ---------------------------------------------------------------- sidebar
SR6_NOTE = ("Yield and net tuition revenue share a factor: expected revenue is "
            "net price weighted by the probability of enrolling. They are not "
            "independent (SR-6).")

# ------------------------------------------------- controls at the top
st.markdown("## Admissions similarity graph")
bar = st.columns([1, 1, 1, 1, 1, 1.15], gap="small")
with bar[0]:
    st.selectbox("Preset", list(cfg.PRESETS), key="preset",
                 on_change=apply_preset,
                 help=cfg.PRESET_DESCRIPTIONS[st.session_state["preset"]])
for col, key in zip(bar[1:5], cfg.PRIORITY_KEYS):
    with col:
        st.select_slider(
            cfg.PRIORITY_LABELS[key], options=cfg.SCALE, key=key,
            help=SR6_NOTE if key in ("w_yield", "w_ntr") else None)
with bar[5]:
    st.select_slider("Detail", options=cfg.DETAIL_ORDER, key="detail_level",
                     help=cfg.DETAIL_DESCRIPTIONS[
                         st.session_state["detail_level"]])

with st.sidebar:
    st.markdown("### Graph")
    st.radio("Group discs by", list(cfg.GROUPINGS), key="group_by",
             help="Position can encode only one grouping, so this is a choice "
                  "rather than a combination.")
    st.caption(cfg.GROUPINGS[st.session_state["group_by"]])
    st.toggle("Group into clusters", key="clustered")
    st.toggle("Show group labels", key="show_cohort_labels")
    st.toggle("Dark graph", key="dark_graph")
    st.selectbox("Graph focus", cfg.FOCUS_MODES, key="focus_mode")

    st.markdown("### Short list")
    st.select_slider("Short list breadth",
                     options=list(cfg.SHORTLIST_SIZE), key="shortlist_size")

    with st.expander("Inside diversity"):
        st.select_slider("Club coverage", options=cfg.SCALE, key="w_club_cov")
        st.caption("Subordinate to the diversity control, and inert when "
                   "diversity is not considered (SR-8c).")
        st.caption("Composition targets are fixed in config.py for this build; "
                   "they are listed in the Composition tab (SR-5c).")

    with st.expander("What counts as alumni success"):
        st.caption("Four definitions, and they select different people. Never "
                   "collapsed into one hidden number (requirements 1.4).")
        for label in priorities.SUCCESS_COMPONENTS:
            st.toggle(label, key=f"succ_{label}")

    with st.expander("Similarity dimensions"):
        for name, _ in cfg.DIMENSIONS:
            st.toggle(name, key=f"dim_on_{name}")
            if st.session_state[f"dim_on_{name}"]:
                st.select_slider(f"{name} importance", options=cfg.SCALE,
                                 key=f"dim_w_{name}",
                                 label_visibility="collapsed")
            if name == "Socioeconomic context":
                st.caption("Off by default. It can serve access goals or "
                           "entrench advantage depending on sign and company "
                           "(FR-2).")

    with st.expander("Matching and clubs"):
        st.select_slider("Match strictness", options=cfg.SCALE,
                         key="match_strictness")
        st.select_slider("Matches shown per applicant",
                         options=list(cfg.MATCHES_PER_APPLICANT),
                         key="matches_per_applicant")
        st.toggle("Show campus clubs", key="show_clubs")
        st.toggle("Surface sole-pipeline applicants", key="show_sole_pipeline")

    # FR-6 requires the mock-data limitations be visible in the app, not only in
    # the requirements document. The tab is gone, so they live here.
    panels.caveats_panel()

    st.toggle("Disclose numeric weights", key="disclose_weights")
    st.button("Reset to defaults", on_click=reset_defaults,
              width="stretch")


# --------------------------------------------------------------- compute
@dataclass
class Computed:
    ds: object
    ctx: object
    M: np.ndarray
    A: np.ndarray
    w: np.ndarray
    sim: np.ndarray
    topk_idx: np.ndarray
    topk_val: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p4: np.ndarray
    thin: np.ndarray
    mask: np.ndarray
    ros: object
    B: np.ndarray
    C: np.ndarray
    target: np.ndarray
    labels: list
    carried: dict
    cost: dict
    priority_words: dict
    dim_words: dict
    priority_weights: dict = field(default_factory=dict)
    cluster_stats: dict = field(default_factory=dict)

    def priority_pct(self, i: int) -> dict:
        return {"w_yield": float(self.p1[i]), "w_ntr": float(self.p2[i]),
                "w_acad": float(self.p4[i]), "w_div": 0.5}

    def config_blob(self) -> dict:
        return {"priorities": self.priority_words,
                "similarity": self.dim_words,
                "advisory": "Discussion set only, not an admission decision."}


def dim_words() -> dict[str, str]:
    return {name: (st.session_state[f"dim_w_{name}"]
                   if st.session_state[f"dim_on_{name}"] else "Not considered")
            for name, _ in cfg.DIMENSIONS}


def priority_words() -> dict[str, str]:
    return {k: st.session_state[k] for k in cfg.PRIORITY_KEYS}


def compute(pw: dict[str, str], dw: dict[str, str] | None = None) -> Computed:
    dw = dw or dim_words()
    C, target, labels = priorities.composition_matrix(
        DS, st.session_state["targets"])
    w = similarity.weights_from_words(dw)
    sim = similarity.contract(boot.M2, boot.A2, w, (DS.n_app, DS.n_alum))

    floor = cfg.STRICTNESS[st.session_state["match_strictness"]]
    k = cfg.MATCHES_PER_APPLICANT[st.session_state["matches_per_applicant"]]
    idx, val = similarity.topk(sim, k, floor)

    enabled = {label: st.session_state[f"succ_{label}"]
               for label in priorities.SUCCESS_COMPONENTS}
    p4, thin, _, _ = priorities.p4_academic(sim, DS, enabled)

    indiv = priorities.blend_individual(
        boot.p1, boot.p2, p4, cfg.WEIGHT[pw["w_yield"]],
        cfg.WEIGHT[pw["w_ntr"]], cfg.WEIGHT[pw["w_acad"]])
    size = round(cfg.CLASS_TARGET
                 * cfg.SHORTLIST_SIZE[st.session_state["shortlist_size"]])
    mask = priorities.build_shortlist(
        indiv, C, target, boot.B, None,
        cfg.WEIGHT[pw["w_div"]], cfg.WEIGHT[st.session_state["w_club_cov"]],
        size, boot.id_rank)

    return Computed(
        ds=DS, ctx=boot.ctx, M=boot.M, A=boot.A, w=w, sim=sim,
        topk_idx=idx, topk_val=val, p1=boot.p1, p2=boot.p2, p4=p4, thin=thin,
        mask=mask, ros=boot.ros, B=boot.B, C=C, target=target,
        labels=labels, carried=boot.carried,
        cost=explain.cost_of_choice(DS, mask),
        priority_words=pw, dim_words=dw,
        priority_weights={k: cfg.WEIGHT[v] for k, v in pw.items()})


PW = priority_words()
C = compute(PW)
boot.ctx.apply_grouping(st.session_state["group_by"],      # free, precomputed
                        st.session_state["clustered"])
boot.ctx.sim = C.sim
boot.ctx.topk_idx, boot.ctx.topk_val = C.topk_idx, C.topk_val
boot.ctx.shortlist = C.mask
boot.ctx.rosters = C.ros


def recompute_mask(words: dict[str, str]) -> np.ndarray:
    return compute(words).mask


# ---- XR-4: attribute the change against the previous configuration
prev = st.session_state.get("prev_priority_words")
if prev and prev != PW:
    st.session_state["last_change"] = explain.attribute_change(
        prev, PW, recompute_mask(prev), C.mask, recompute_mask, DS)
st.session_state["prev_priority_words"] = dict(PW)

# ---- XR-8: counterfactual for the selected applicant only, on demand
sel = st.session_state.get("last_clicked")
if sel in DS.app_index:
    i = DS.app_index[sel]
    st.session_state["counterfactuals"] = {
        sel: explain.counterfactual(i, PW, recompute_mask, bool(C.mask[i]))}


# ---- per-cohort short list counts and the change since the last setting.
# This is what makes a slider's effect on a group visible (the cluster labels
# read "31 of 78 shortlisted  ▲6").
def cohort_stats(mask: np.ndarray, prev: np.ndarray | None) -> dict:
    a = boot.ctx.assignment
    out: dict[str, tuple[int, int, int]] = {}
    if a is None:
        return out
    for cohort, ids in a.members.items():
        idx = [DS.app_index[p] for p in ids if p in DS.app_index]
        if not idx:
            continue
        picked = int(mask[idx].sum())
        before = int(prev[idx].sum()) if prev is not None else picked
        out[cohort] = (picked, len(idx), picked - before)
    return out


STATS = cohort_stats(C.mask, st.session_state.get("prev_mask"))
st.session_state["prev_mask"] = C.mask.copy()

# XR-9 / XR-13: always on screen, never numeric.
st.caption(explain.config_sentence(
    PW, st.session_state["shortlist_size"], st.session_state["w_club_cov"],
    [n for n, word in C.dim_words.items() if word == "Not considered"],
    detail=st.session_state["detail_level"],
    clustered=st.session_state["clustered"])
    + ("  ·  Ethnicity and sex excluded from scoring (FR-5)."
       if all(boot.audit.values()) else "  ·  SUPPRESSION AUDIT FAILED."))

if st.session_state["show_sole_pipeline"]:
    extra = [p for p in boot.carried
             if not C.mask[DS.app_index[p]]]
    if extra:
        st.warning(
            f"{len(extra)} applicants outside this short list are the sole "
            f"realistic pipeline for a club at succession risk: "
            f"{', '.join(extra[:8])}"
            f"{'…' if len(extra) > 8 else ''}. None of the four priorities can "
            f"express this, which is why it is surfaced separately rather than "
            f"folded into the ranking (IR-7e).")

C.cluster_stats = STATS

tabs = st.tabs(["Graph", "Short list", "Why", "Composition", "What changed"])
with tabs[0]:
    panels.graph_tab(C)
with tabs[1]:
    panels.shortlist_tab(C)
with tabs[2]:
    panels.why_tab(C)
with tabs[3]:
    panels.composition_tab(C)
with tabs[4]:
    panels.changed_tab(C)
