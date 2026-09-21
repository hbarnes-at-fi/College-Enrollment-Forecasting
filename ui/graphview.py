"""Plotly figure assembly (specs §9).

Performance rests on one rule: one trace per edge tier, not one trace per edge.
Coordinates are concatenated with `None` separators so an entire tier is a single
trace. Nine traces total regardless of graph size.

Shape carries node type independently of colour (NFR-3); colour alone fails
colour-blind users.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg                                  # noqa: E402
from core.graphbuild import Edge, GraphContext        # noqa: E402

TIER_ORDER = ["observed_past", "observed_intent", "inferred"]
KIND_ORDER = ["Activity", "Athletic team", "Major", "Campus club",
              "Alum", "Applicant"]

HIGHLIGHT = "#1A1A1A"
DIM_OPACITY = 0.22
CLUB_SIZE = {"Secure": 9, "Thin": 12, "At risk": 16, "Critical": 19}


def _roster_lookup(ctx: GraphContext) -> dict[str, tuple[str, bool]]:
    if ctx.rosters is None:
        return {}
    return {r.club_id: (r.band, bool(r.at_risk))
            for r in ctx.rosters.itertuples(index=False)}


TIER_LABEL = {"observed_past": "Observed history",
              "observed_intent": "Stated intent",
              "inferred": "Inferred similarity"}


def _segments(pairs: list[tuple[str, str]], lay: dict,
              pos: dict[str, int], X: np.ndarray, Y: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised line segments with NaN separators.

    Plotly validates Python lists element by element: profiling the 755-node
    default view showed 242,010 calls into `to_scalar_or_list`, which was most
    of the figure build. numpy arrays skip that path, and NaN breaks a line the
    same way None does.
    """
    if not pairs:
        return np.empty(0), np.empty(0)
    si = np.fromiter((pos[a] for a, _ in pairs), dtype=np.int32, count=len(pairs))
    ti = np.fromiter((pos[b] for _, b in pairs), dtype=np.int32, count=len(pairs))
    xs = np.full(len(pairs) * 3, np.nan)
    ys = np.full(len(pairs) * 3, np.nan)
    xs[0::3], xs[1::3] = X[si], X[ti]
    ys[0::3], ys[1::3] = Y[si], Y[ti]
    return xs, ys


def _filter_edges(edges: list[Edge], spec: dict, focused: bool,
                  emphasis: set[tuple[str, str]] | None = None) -> list[Edge]:
    """NR-5a to NR-5c, with NR-5d taking precedence over all of them.

    `SIMILAR_TO` is inferred, so a wide view has no business asserting it. But if
    a user has asked why two people are connected, the answer is drawn at every
    detail level including `Clusters only` — an explanation that disappears
    because of a display setting is not an explanation.
    """
    def emphasised(e: Edge) -> bool:
        return bool(emphasis) and ((e.source, e.target) in emphasis
                                   or (e.target, e.source) in emphasis)

    if not spec["edges"]:
        return [e for e in edges if emphasised(e)]

    rule = spec["similar"]
    if rule == "always":
        return edges
    allow_similar = rule == "focus" and focused
    return [e for e in edges
            if e.kind != "SIMILAR_TO" or allow_similar or emphasised(e)]


def _edge_traces(ctx: GraphContext, edges: list[Edge],
                 emphasis: set[tuple[str, str]] | None,
                 base_opacity: float = 1.0,
                 pal: dict | None = None) -> list[go.Scattergl]:
    pal = pal or cfg.theme()
    lay = ctx.layout
    pos = ctx._pos_index
    X, Y = ctx._pos_x, ctx._pos_y

    cold: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_ORDER}
    hot: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_ORDER}
    for e in edges:
        if e.source not in pos or e.target not in pos:
            continue
        is_hot = bool(emphasis) and ((e.source, e.target) in emphasis
                                    or (e.target, e.source) in emphasis)
        bucket = hot if is_hot else cold
        bucket.setdefault(e.tier, []).append((e.source, e.target))

    traces: list[go.Scattergl] = []
    for tier in TIER_ORDER:
        dash = cfg.EDGE_DASHES[tier]
        width = cfg.EDGE_WIDTHS[tier]
        colour = pal["edge"][tier]
        label = TIER_LABEL[tier]
        if cold.get(tier):
            xs, ys = _segments(cold[tier], lay, pos, X, Y)
            opacity = base_opacity * (DIM_OPACITY if emphasis else 1.0)
            traces.append(go.Scattergl(
                x=xs, y=ys, mode="lines", name=label, hoverinfo="skip",
                legendgroup=tier,
                opacity=max(0.02, min(1.0, opacity)),
                line=dict(width=width, color=colour, dash=dash)))
        if hot.get(tier):
            xs, ys = _segments(hot[tier], lay, pos, X, Y)
            traces.append(go.Scattergl(
                x=xs, y=ys, mode="lines", name=f"{label} (shared)",
                hoverinfo="skip", legendgroup=tier, showlegend=False,
                line=dict(width=width + 1.8, color=pal["highlight"],
                          dash=dash)))
    return traces


def _hover(ctx: GraphContext, nid: str, rosters: dict) -> str:
    n = ctx.nodes[nid]
    m = n.meta
    if n.kind == "Applicant":
        return (f"<b>{n.label}</b><br>{m['major']}<br>{m['region']}"
                f"<br>Reader rating {m['rating']}"
                f"<br>{m['income']}"
                f"<br>First-generation: {m['first_gen']}")
    if n.kind == "Alum":
        return (f"<b>{n.label}</b><br>{m['major']}<br>Class of {m['cohort']}"
                f"<br>{m['tier']}"
                + ("<br><i>Exemplar</i>" if m.get("exemplar") else ""))
    if n.kind == "Campus club":
        band, risk = rosters.get(nid, ("unknown", False))
        tail = "<br><b>At succession risk</b>" if risk else ""
        proxy = ("<br><i>Excluded from scoring: identity proxy</i>"
                 if m.get("proxy_sensitive") else "")
        return f"<b>{n.label}</b><br>Roster: {band}{tail}{proxy}"
    return f"<b>{n.label}</b><br>{n.kind}"


def _node_traces(ctx: GraphContext, keep: set[str],
                 highlight: set[str] | None,
                 node_scale: float = 1.0,
                 pal: dict | None = None) -> list[go.Scattergl]:
    pal = pal or cfg.theme()
    lay = ctx.layout
    pos, X, Y = ctx._pos_index, ctx._pos_x, ctx._pos_y
    rosters = _roster_lookup(ctx)
    shortlisted = set()
    if ctx.shortlist is not None:
        shortlisted = {ctx.ds.app_ids[i] for i in np.flatnonzero(ctx.shortlist)}

    by_kind: dict[str, list[str]] = {}
    for nid in keep:
        if nid in lay:
            by_kind.setdefault(ctx.nodes[nid].kind, []).append(nid)

    traces: list[go.Scattergl] = []
    for kind in KIND_ORDER:
        ids = sorted(by_kind.get(kind) or [])
        if not ids:
            continue
        symbol = cfg.NODE_SYMBOLS[kind]
        base_size = cfg.NODE_SIZES[kind]
        base_colour = pal["node"][kind]
        sizes, colours, widths, lines, opac = [], [], [], [], []
        for nid in ids:
            meta = ctx.nodes[nid].meta
            size = base_size
            colour = base_colour
            lw, lc, op = 0.4, pal["plot"], 1.0
            if kind == "Campus club":
                band, risk = rosters.get(nid, ("Secure", False))
                size = CLUB_SIZE.get(band, base_size)
                if risk:
                    lw, lc = 2.6, "#E0553C"
            elif kind == "Alum" and meta.get("exemplar"):
                lw, lc = 2.0, pal["node"]["Alum"]
            elif kind == "Applicant":
                if nid in shortlisted:
                    # the loudest signal available, because this is the thing a
                    # slider moves and it has to be visible across 755 nodes
                    lw = cfg.SHORTLIST_OUTLINE_WIDTH
                    lc = cfg.SHORTLIST_OUTLINE
                    size = base_size + 1.0
                else:
                    colour = pal["node_dim"]
                    op = cfg.UNPICKED_OPACITY
            sizes.append(size * node_scale)
            colours.append(colour)
            widths.append(lw)
            lines.append(lc)
            opac.append(op)

        opacity = 1.0
        if highlight:
            opacity = 1.0 if any(i in highlight for i in ids) else DIM_OPACITY

        # Applicants split into two traces so the legend names the short list
        # instead of leaving a yellow ring unexplained. One extra trace against
        # the specs §9 count, which is worth it for a self-documenting legend.
        groups: list[tuple[str, list[int]]]
        if kind == "Applicant" and shortlisted:
            on = [i for i, nid in enumerate(ids) if nid in shortlisted]
            off = [i for i, nid in enumerate(ids) if nid not in shortlisted]
            groups = [("Applicant, not on short list", off),
                      ("Applicant, on short list", on)]
        else:
            groups = [(kind, list(range(len(ids))))]

        for label, picks in groups:
            if not picks:
                continue
            gids = [ids[i] for i in picks]
            sel = np.fromiter((pos[i] for i in gids), dtype=np.int32,
                              count=len(gids))
            traces.append(go.Scattergl(
                x=X[sel], y=Y[sel],
                mode="markers", name=label, customdata=gids,
                text=[_hover(ctx, i, rosters) for i in gids],
                hovertemplate="%{text}<extra></extra>",
                opacity=opacity * min(opac[i] for i in picks),
                marker=dict(symbol=symbol,
                            size=np.asarray([sizes[i] for i in picks],
                                            dtype=np.float32),
                            color=[colours[i] for i in picks],
                            line=dict(width=np.asarray(
                                [widths[i] for i in picks], dtype=np.float32),
                                color=[lines[i] for i in picks]))))

    if highlight:
        hot = sorted(i for i in highlight if i in pos)
        if hot:
            sel = np.fromiter((pos[i] for i in hot), dtype=np.int32, count=len(hot))
            traces.append(go.Scattergl(
                x=X[sel], y=Y[sel],
                mode="markers", name="Shared evidence", customdata=hot,
                text=[_hover(ctx, i, rosters) for i in hot],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(symbol="circle-open", size=20,
                            color=pal["highlight"],
                            line=dict(width=2.4, color=pal["highlight"]))))
    return traces


def _cluster_chrome(ctx: GraphContext, keep: set[str], labels: bool,
                    isolate: str | None, pal: dict,
                    stats: dict | None) -> tuple[list[dict], list[dict]]:
    """NR-4 cluster discs and labels, as layout shapes and annotations.

    Layout-level rather than traces, so specs §9's nine-trace contract is
    unaffected and the cost is constant in graph size.

    `stats` carries per-cohort short-list counts and the change since the last
    configuration. That is where the effect of moving a slider is reported. It
    goes in the LABEL rather than into the disc fill deliberately: brightening a
    disc would make some clusters look better than others, which FR-8 forbids,
    and a number that visibly ticks up or down answers the question more exactly
    anyway.
    """
    if not ctx.clustered or not ctx.cluster_geo:
        return [], []

    present: dict[str, int] = {}
    for nid in keep:
        c = ctx.nodes[nid].meta.get("cohort")
        if c:
            present[c] = present.get(c, 0) + 1

    shapes: list[dict] = []
    notes: list[dict] = []
    for cohort, g in ctx.cluster_geo.items():
        if cohort not in present:
            continue
        dim = isolate is not None and cohort != isolate
        cx, cy = g["centre"]
        r = g["radius"] * 1.18
        shapes.append(dict(
            type="circle", xref="x", yref="y",
            x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
            fillcolor=pal["cluster_fill"] if not dim else "rgba(0,0,0,0.02)",
            line=dict(color=pal["cluster_line"] if not dim
                      else "rgba(0,0,0,0.05)", width=1),
            layer="below"))
        if not labels:
            continue

        line2 = f"{present[cohort]} shown"
        if stats and cohort in stats:
            picked, total, delta = stats[cohort]
            arrow = ""
            if delta > 0:
                arrow = f"  <span style='color:{cfg.SHORTLIST_OUTLINE}'>" \
                        f"▲{delta}</span>"
            elif delta < 0:
                arrow = f"  <span style='color:#E2725B'>▼{abs(delta)}</span>"
            line2 = (f"<span style='color:{cfg.SHORTLIST_OUTLINE}'>{picked}"
                     f"</span> of {total} shortlisted{arrow}")
        notes.append(dict(
            x=cx, y=cy + r + 0.11, xref="x", yref="y",
            text=f"<b>{cohort}</b><br>{line2}",
            showarrow=False, align="center",
            font=dict(size=12 if not dim else 10, color=pal["label"]),
            opacity=0.35 if dim else 1.0))
    return shapes, notes


def _match_traces(ctx: GraphContext, match: dict, pal: dict) -> list[go.Scattergl]:
    """The answer layer: one selected applicant and the past students it matches.

    Drawn last so it sits on top, and given its own legend entries so nothing on
    screen is an unexplained convention.
    """
    pos, X, Y = ctx._pos_index, ctx._pos_x, ctx._pos_y
    rosters = _roster_lookup(ctx)
    app_id = match["applicant"]
    pairs = [(a, s) for a, s in match["alumni"] if a in pos]
    traces: list[go.Scattergl] = []

    # bright connectors, width by similarity so rank is visible at a glance
    if pairs and app_id in pos:
        for rank, (alum_id, sim) in enumerate(pairs):
            x0, y0 = X[pos[app_id]], Y[pos[app_id]]
            x1, y1 = X[pos[alum_id]], Y[pos[alum_id]]
            traces.append(go.Scattergl(
                x=np.array([x0, x1]), y=np.array([y0, y1]),
                mode="lines", hoverinfo="skip",
                # every connector carries the same name so the layer is
                # identifiable; only the first claims a legend slot
                name="Match", legendgroup="match",
                showlegend=(rank == 0),
                line=dict(width=1.2 + 3.4 * float(sim), color=cfg.FOCUS_EDGE),
                opacity=0.95))

    if pairs:
        ids = [a for a, _ in pairs]
        sel = np.fromiter((pos[i] for i in ids), dtype=np.int32, count=len(ids))
        labels = []
        for rank, (alum_id, sim) in enumerate(pairs):
            rl = (cfg.FOCUS_RANK_LABELS[rank]
                  if rank < len(cfg.FOCUS_RANK_LABELS) else f"#{rank + 1}")
            row = ctx.ds.alumni.iloc[ctx.ds.alum_index[alum_id]]
            labels.append(
                f"<b>{alum_id}</b> — {rl} match<br>"
                f"{row.final_major}, class of {row.cohort_year}<br>"
                f"{row.success_composite_tier}<br>"
                f"Final GPA {float(row.final_cum_gpa):.2f}")
        traces.append(go.Scattergl(
            x=X[sel], y=Y[sel], mode="markers",
            name="Similar past students", customdata=ids,
            text=labels, hovertemplate="%{text}<extra></extra>",
            marker=dict(symbol=cfg.NODE_SYMBOLS["Alum"],
                        size=cfg.FOCUS_ALUM_SIZE,
                        color=pal["node"]["Alum"],
                        line=dict(width=3.0, color=cfg.FOCUS_ALUM_RING))))

    if app_id in pos:
        i = pos[app_id]
        traces.append(go.Scattergl(
            x=X[[i]], y=Y[[i]], mode="markers",
            name="Selected applicant", customdata=[app_id],
            text=[_hover(ctx, app_id, rosters)],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(symbol=cfg.NODE_SYMBOLS["Applicant"],
                        size=cfg.FOCUS_NODE_SIZE,
                        color=pal["node"]["Applicant"],
                        line=dict(width=3.4, color=cfg.FOCUS_APPLICANT_RING))))
    return traces


def build_figure(ctx: GraphContext, keep: set[str], edges: list[Edge],
                 highlight: set[str] | None = None,
                 emphasis: set[tuple[str, str]] | None = None,
                 height: int = 640,
                 detail: str = cfg.DETAIL_DEFAULT,
                 show_labels: bool = True,
                 isolate: str | None = None,
                 focused: bool = False,
                 theme: str = cfg.DEFAULT_THEME,
                 cluster_stats: dict | None = None,
                 match_focus: dict | None = None) -> go.Figure:
    """XR-6: `highlight` and `emphasis` carry the shared nodes and shared edges
    of a selected pair, so the evidence is visible in the graph rather than
    asserted in a tooltip.

    NR-5: `detail` gates which edges are drawn and how faintly. Opacity is the
    mechanism that makes the view zoom-responsive with no events at all; at wide
    view thousands of faint edges overlap into a wash, and the same edges resolve
    into individual lines once panned and zoomed into a region.
    """
    pal = cfg.theme(theme)
    spec = cfg.DETAIL_LEVELS.get(detail, cfg.DETAIL_LEVELS[cfg.DETAIL_DEFAULT])
    drawn = _filter_edges(edges, spec, focused, emphasis)

    # With one applicant selected, the rest of the graph becomes context rather
    # than content. Pushing it right back is the only way to make the match
    # unmistakable, because the frozen layout cannot move the alumni closer.
    backdrop = cfg.FOCUS_BACKDROP_OPACITY if match_focus else None

    fig = go.Figure()
    for t in _edge_traces(ctx, drawn, emphasis, spec["edge_opacity"], pal):
        if backdrop is not None:
            t.opacity = min(t.opacity or 1.0, backdrop)
            t.showlegend = False
        fig.add_trace(t)
    for t in _node_traces(ctx, keep, highlight, spec["node_scale"], pal):
        if backdrop is not None:
            t.opacity = min(t.opacity or 1.0, backdrop)
            t.showlegend = False
        fig.add_trace(t)

    if match_focus:
        for t in _match_traces(ctx, match_focus, pal):
            fig.add_trace(t)

    shapes, notes = _cluster_chrome(
        ctx, keep, labels=show_labels and spec["labels"], isolate=isolate,
        pal=pal, stats=cluster_stats)
    if match_focus:
        # labels stay, but stop competing with the answer
        for n in notes:
            n["opacity"] = 0.30

    fig.update_layout(
        shapes=shapes, annotations=notes,
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11, color=pal["text"])),
        xaxis=dict(visible=False, fixedrange=False),
        yaxis=dict(visible=False, fixedrange=False,
                   scaleanchor="x", scaleratio=1.0),
        plot_bgcolor=pal["plot"],
        paper_bgcolor=pal["paper"],
        font=dict(color=pal["text"]),
        dragmode="pan",
        hoverlabel=dict(align="left"),
        uirevision="frozen",   # NFR-2: keep pan/zoom across reruns
    )
    return fig


def selected_ids(event) -> list[str]:
    """Pull node ids out of a Streamlit plotly selection event."""
    if not event:
        return []
    points = (event.get("selection", {}) or {}).get("points", []) or []
    out: list[str] = []
    for p in points:
        cd = p.get("customdata")
        if isinstance(cd, list):
            cd = cd[0] if cd else None
        if isinstance(cd, str):
            out.append(cd)
    return out


def detail_caption(detail: str, clustered: bool,
                   grouping: str | None = None) -> str:
    """XR-13: a view with no visible edges must never be ambiguous between
    'no edges exist' and 'edges are not being drawn'."""
    text = cfg.DETAIL_DESCRIPTIONS.get(detail, "")
    if clustered:
        what = (grouping or "Cohort").lower()
        text += (f" Discs group people by {what}; ring position carries no "
                 f"ranking.")
    else:
        text += (" Clustering is off: applicants and past students sit on "
                 "opposing arcs.")
    return text


def legend_caption() -> str:
    """NFR-3: every visual claim must also exist in words.

    Shape was doing too much work here. Size and hue now carry the population
    distinction, with shape kept as the third cue rather than the only one.
    """
    return (
        "Applicants are the small blue dots. Past students are the larger "
        "violet squares, with a heavier outline when they are exemplars. "
        "A yellow ring means an applicant is on the current short list. "
        "Grey diamonds are activities, triangles are teams, gold stars are "
        "campus clubs sized by roster health with a red outline when at "
        "succession risk. "
        "Each person shows only their three most informative connections: "
        "solid is observed history, dashed is stated intent, dotted is "
        "inferred similarity."
    )
