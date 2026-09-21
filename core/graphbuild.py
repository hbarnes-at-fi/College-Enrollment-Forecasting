"""Nodes, observed edges, frozen layout and subgraph extraction (NR-1 to NR-3).

The layout is computed once, over observed edges only, and never recomputed.
`SIMILAR_TO` is excluded from layout entirely because it changes with every
weight adjustment; including it would move every node on every interaction and
defeat XR-4, which depends on the user being able to see what moved.

The arrangement is deterministic and layered rather than force-directed. At 1,925
nodes a spring layout produces an undifferentiated hairball, and it would also be
the slowest thing in the build.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg               # noqa: E402
from core.loader import Dataset    # noqa: E402

# central bands, y position per attribute kind
BANDS = {"Major": 0.72, "Activity": 0.24, "Athletic team": -0.24,
         "Campus club": -0.72}
BAND_HALF_WIDTH = 0.62
APP_RINGS = [1.30, 1.48, 1.66, 1.84]
ALUM_RINGS = [1.30, 1.48, 1.66]
APP_ARC = (100.0, 260.0)      # degrees, left side
ALUM_ARC = (-80.0, 80.0)      # right side


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str

    @property
    def tier(self) -> str:
        if self.kind in cfg.OBSERVED_PAST_EDGES:
            return "observed_past"
        if self.kind in cfg.OBSERVED_INTENT_EDGES:
            return "observed_intent"
        return "inferred"


@dataclass
class GraphContext:
    ds: Dataset
    nodes: dict[str, Node]
    edges: list[Edge]
    layout: dict[str, tuple[float, float]]
    adjacency: dict[str, set[str]]
    sim: np.ndarray | None = None
    topk_idx: np.ndarray | None = None
    topk_val: np.ndarray | None = None
    shortlist: np.ndarray | None = None
    rosters: object | None = None

    # NR-4: both arrangements are precomputed and frozen. Switching must not
    # trigger a recompute, and cohort assignment depends only on loaded data
    # (NFR-6), never on a control value.
    assignment: object | None = None
    cluster_geo: dict = field(default_factory=dict)
    layouts: dict[str, dict] = field(default_factory=dict)
    clustered: bool = True
    assignments: dict[str, object] = field(default_factory=dict)
    geometries: dict[str, dict] = field(default_factory=dict)
    grouping: str = cfg.GROUP_BY_DEFAULT

    # position lookup as arrays, so the renderer can hand numpy straight to
    # Plotly instead of Python lists it would validate element by element
    _pos_index: dict[str, int] = field(default_factory=dict)
    _pos_x: np.ndarray | None = None
    _pos_y: np.ndarray | None = None
    _person_links: list | None = None

    def index_positions(self) -> None:
        ids = list(self.layout)
        self._pos_index = {nid: i for i, nid in enumerate(ids)}
        self._pos_x = np.array([self.layout[i][0] for i in ids], dtype=np.float64)
        self._pos_y = np.array([self.layout[i][1] for i in ids], dtype=np.float64)

    def apply_grouping(self, grouping: str, clustered: bool = True) -> None:
        """Switch which attribute the discs represent. Free: all precomputed."""
        if grouping not in self.assignments:
            grouping = cfg.GROUP_BY_DEFAULT
        self.grouping = grouping
        self.clustered = clustered
        self.assignment = self.assignments[grouping]
        self.cluster_geo = self.geometries[grouping]
        self.layout = self.layouts[grouping if clustered else "arcs"]
        # the group a node sits in travels with the node, so the renderer and
        # panels never re-derive it
        for nid, node in self.nodes.items():
            primary = self.assignment.primary.get(nid)
            node.meta["cohort"] = primary
            node.meta["cohorts"] = self.assignment.memberships.get(nid, [])
        self.index_positions()

    def set_arrangement(self, clustered: bool) -> None:
        """IR-8d. Cheap: every layout is already built."""
        self.apply_grouping(self.grouping, clustered)

    def person_links(self) -> list:
        """Cached: depends only on loaded data, never on a control value."""
        if self._person_links is None:
            from core.graphbuild import person_links
            self._person_links = person_links(self)
        return self._person_links



    def cohort_of(self, nid: str) -> str | None:
        if self.assignment is None:
            return None
        return self.assignment.primary.get(nid)

    def cohorts_of(self, nid: str) -> list[str]:
        if self.assignment is None:
            return []
        return list(self.assignment.memberships.get(nid, []))


# --------------------------------------------------------------- nodes
def _activity_names(ds: Dataset, pid: str, athletic: set[str]) -> list[str]:
    """Activities as a characteristic, rarest first so the informative ones lead."""
    ids = [a for a in ds.act_sets.get(pid, set()) if a not in athletic]
    ids.sort(key=lambda a: -ds.act_idf.get(a, 0.0))
    return [ds.activity_name(a) for a in ids]


def _club_names(ds: Dataset, pid: str) -> list[str]:
    """Clubs as a characteristic. `proxy_sensitive` clubs are excluded from the
    keyword list as well as from scoring: displaying an affinity or religious
    club as a badge on a person is the same inference §0.1 forbids."""
    out = []
    for cid in sorted(ds.club_sets.get(pid, set())):
        if cid in ds.proxy_clubs:
            continue
        out.append(ds.club_name(cid))
    return out


def build_nodes(ds: Dataset) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    athletic = set(ds.activities.loc[ds.activities.category == "Athletics",
                                     "activity_id"])

    for r in ds.applicants.itertuples(index=False):
        sport = str(r.athletic_recruit_sport or "").strip()
        sport = "" if sport.lower() in ("nan", "none") else sport
        nodes[r.applicant_id] = Node(
            r.applicant_id, "Applicant", r.applicant_id,
            {"school": r.intended_school, "major": r.intended_major,
             "region": r.region, "rating": int(r.academic_rating),
             "yield": float(r.predicted_yield_prob),
             "net_price": float(r.net_price),
             "first_gen": r.first_generation,
             "income": r.family_income_band,
             "disposition": r.committee_disposition,
             # characteristics, formerly node types of their own
             "sport": sport,
             "athlete": bool(sport),
             "clubs": _club_names(ds, r.applicant_id),
             "activities": _activity_names(ds, r.applicant_id, athletic)})

    for r in ds.alumni.itertuples(index=False):
        sport = str(r.entry_athletic_recruit_sport or "").strip()
        sport = "" if sport.lower() in ("nan", "none") else sport
        nodes[r.alum_id] = Node(
            r.alum_id, "Alum", r.alum_id,
            {"school": r.entry_intended_school, "major": r.final_major,
             "region": r.entry_region, "rating": int(r.entry_academic_rating),
             "cohort_year": int(r.cohort_year), "gpa": float(r.final_cum_gpa),
             "tier": r.success_composite_tier,
             "exemplar": r.exemplar_flag == "Y",
             "sport": sport,
             "athlete": bool(sport) or int(r.varsity_seasons) > 0,
             "seasons": int(r.varsity_seasons),
             "captain": str(r.team_captain) == "Y",
             "clubs": _club_names(ds, r.alum_id),
             "activities": _activity_names(ds, r.alum_id, athletic)})

    # §0.2: sports are activities_dim rows, rendered as Athletic team. One node
    # per sport, not two.
    for r in ds.activities.itertuples(index=False):
        kind = "Athletic team" if r.category == "Athletics" else "Activity"
        nodes[r.activity_id] = Node(r.activity_id, kind, r.activity_name,
                                    {"category": r.category})

    majors = sorted(set(ds.applicants.intended_major)
                    | set(ds.alumni.final_major)
                    | set(ds.alumni.entry_intended_major))
    for m in majors:
        nodes[f"MAJ::{m}"] = Node(f"MAJ::{m}", "Major", m, {})

    for r in ds.clubs.itertuples(index=False):
        nodes[r.club_id] = Node(
            r.club_id, "Campus club", r.club_name,
            {"category": r.category,
             "linked_activity_id": str(r.linked_activity_id or ""),
             "proxy_sensitive": r.proxy_sensitive == "Y",
             "min_viable": int(r.min_viable_members)})
    return nodes


# --------------------------------------------------------------- edges
def build_observed_edges(ds: Dataset) -> list[Edge]:
    kind_of = dict(zip(ds.activities.activity_id, ds.activities.category))
    edges: list[Edge] = []

    for r in ds.act_edges.itertuples(index=False):
        athletic = kind_of.get(r.activity_id) == "Athletics"
        edges.append(Edge(r.person_id, r.activity_id,
                          "PLAYS_FOR" if athletic else "PARTICIPATES_IN"))

    for r in ds.applicants.itertuples(index=False):
        edges.append(Edge(r.applicant_id, f"MAJ::{r.intended_major}", "STUDIES"))
    for r in ds.alumni.itertuples(index=False):
        edges.append(Edge(r.alum_id, f"MAJ::{r.final_major}", "STUDIES"))

    for r in ds.club_edges.itertuples(index=False):
        edges.append(Edge(r.person_id, r.club_id, r.relationship))

    # dedupe; a person can hold several roles in one activity
    seen: set[tuple[str, str, str]] = set()
    unique: list[Edge] = []
    for e in edges:
        key = (e.source, e.target, e.kind)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


# Edges whose endpoint is now a characteristic rather than a node worth drawing.
# Adjacency still holds them, because XR-6 shared-path highlighting and the club
# panels need to know a person studies X or belongs to Y.
CHARACTERISTIC_EDGES = {"STUDIES", "PLAYS_FOR", "MEMBER_OF", "INTERESTED_IN"}


def build_adjacency(edges: list[Edge]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for e in edges:
        adj.setdefault(e.source, set()).add(e.target)
        adj.setdefault(e.target, set()).add(e.source)
    return adj


# --------------------------------------------------------------- layout
def _arc(ids: list[str], rings: list[float], arc: tuple[float, float]
         ) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    per = math.ceil(len(ids) / len(rings)) or 1
    a0, a1 = arc
    for r_i, radius in enumerate(rings):
        chunk = ids[r_i * per:(r_i + 1) * per]
        if not chunk:
            continue
        for j, nid in enumerate(chunk):
            frac = j / max(1, len(chunk) - 1)
            ang = math.radians(a0 + (a1 - a0) * frac)
            out[nid] = (radius * math.cos(ang), radius * math.sin(ang) * 0.85)
    return out


def _band(ids: list[str], y: float) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    n = max(1, len(ids) - 1)
    for i, nid in enumerate(ids):
        x = -BAND_HALF_WIDTH + 2 * BAND_HALF_WIDTH * (i / n)
        # gentle zig so labels do not collide on a dense band
        out[nid] = (x, y + (0.055 if i % 2 else -0.055))
    return out


def freeze_layout(nodes: dict[str, Node], ds: Dataset
                  ) -> dict[str, tuple[float, float]]:
    """Arrangement B, population arcs. Retained under IR-8d.

    Shows connectivity well and structure poorly, which is why NR-4 exists. Kept
    because clustering imposes one reading of the pool and a user must be able to
    remove it.
    """
    layout: dict[str, tuple[float, float]] = {}

    def sort_people(kind: str) -> list[str]:
        people = [n for n in nodes.values() if n.kind == kind]
        return [n.id for n in sorted(
            people, key=lambda n: (n.meta.get("school", ""),
                                   n.meta.get("region", ""), n.id))]

    layout.update(_arc(sort_people("Applicant"), APP_RINGS, APP_ARC))
    layout.update(_arc(sort_people("Alum"), ALUM_RINGS, ALUM_ARC))

    for kind, y in BANDS.items():
        ids = sorted((n.id for n in nodes.values() if n.kind == kind),
                     key=lambda i: (nodes[i].meta.get("category", ""),
                                    nodes[i].label))
        layout.update(_band(ids, y))

    missing = set(nodes) - set(layout)
    if missing:
        raise AssertionError(f"{len(missing)} nodes have no position")
    return layout


# ------------------------------------------------ NR-4 clustered layout
def _phyllotaxis(n: int, radius: float) -> list[tuple[float, float]]:
    """Even fill of a disc with no random seed.

    r = R*sqrt((i+0.5)/n), theta = i * golden angle. Deterministic, and it avoids
    the clumping that uniform random sampling in a disc produces.
    """
    out = []
    for i in range(n):
        r = radius * math.sqrt((i + 0.5) / max(1, n))
        ang = math.radians(cfg.GOLDEN_ANGLE * i)
        out.append((r * math.cos(ang), r * math.sin(ang)))
    return out


def cluster_geometry(assignment) -> dict[str, dict]:
    """Disc centre and radius per cohort, in the computed precedence order.

    A ring has no beginning, so position around it encodes no rank. That is the
    cheapest available way to satisfy FR-8; a left-to-right row would have
    implied an ordering.
    """
    used = [c for c in assignment.order if c in assignment.members]
    n = max(1, len(used))
    biggest = max((len(assignment.members[c]) for c in used), default=1)

    # Shrink discs when there are many groups, then grow the ring to fit them.
    # Grouping by major produces roughly three times as many discs as grouping by
    # cohort, and a fixed ring would overlap them.
    disc_max = cfg.CLUSTER_DISC_MAX if n <= 14 else max(
        cfg.CLUSTER_DISC_MIN + 0.06, cfg.CLUSTER_DISC_MAX * 14.0 / n)
    needed = (2 * disc_max + cfg.CLUSTER_RING_MARGIN) / (
        2 * math.sin(math.pi / n)) if n > 1 else 0.0
    ring = max(cfg.CLUSTER_RING_RADIUS, needed)

    geo: dict[str, dict] = {}
    for i, cohort in enumerate(used):
        ang = 2 * math.pi * i / n - math.pi / 2
        count = len(assignment.members[cohort])
        # radius scales with sqrt(count): fixed discs would make large cohorts
        # unreadably dense and small ones look insignificant, implying a rank
        frac = math.sqrt(count / max(1, biggest))
        radius = cfg.CLUSTER_DISC_MIN + frac * (disc_max - cfg.CLUSTER_DISC_MIN)
        geo[cohort] = {
            "centre": (ring * math.cos(ang), ring * math.sin(ang)),
            "radius": radius,
            "angle": ang,
            "count": count,
            "index": i,
            "ring": ring,
        }
    return geo


# --------------------------- person-to-person links, NR-5 top three
def _shared_matrix(ds: Dataset, people: list[str]) -> tuple[np.ndarray, list[str]]:
    """Person x characteristic, weighted by rarity times kind weight.

    Everything a person carries becomes a column: activities, sports, clubs and
    major. Rarity comes from the same idf the similarity dimensions use, so a
    shared Quiz Bowl counts for more than a shared Community Volunteering.
    """
    sport_of = {}
    for r in ds.applicants.itertuples(index=False):
        s = str(r.athletic_recruit_sport or "").strip()
        sport_of[r.applicant_id] = "" if s.lower() in ("nan", "none") else s
    for r in ds.alumni.itertuples(index=False):
        s = str(r.entry_athletic_recruit_sport or "").strip()
        sport_of[r.alum_id] = "" if s.lower() in ("nan", "none") else s
    major_of = dict(zip(ds.applicants.applicant_id, ds.applicants.intended_major))
    major_of.update(zip(ds.alumni.alum_id, ds.alumni.final_major))

    athletic = set(ds.activities.loc[ds.activities.category == "Athletics",
                                     "activity_id"])
    cols: dict[str, int] = {}

    def col(key: str) -> int:
        return cols.setdefault(key, len(cols))

    entries: list[tuple[int, int, float]] = []
    n_people = len(people)
    majors_count: dict[str, int] = {}
    for p in people:
        majors_count[str(major_of.get(p, ""))] = \
            majors_count.get(str(major_of.get(p, "")), 0) + 1

    for i, p in enumerate(people):
        for aid in ds.act_sets.get(p, ()):  # noqa: SIM118
            kind = "sport" if aid in athletic else "activity"
            w = ds.act_idf.get(aid, 1.0) * cfg.SHARED_KIND_WEIGHT[kind]
            entries.append((i, col(f"a:{aid}"), w))
        for cid in ds.club_sets.get(p, ()):  # noqa: SIM118
            if cid in ds.proxy_clubs:
                continue          # §0.1: never a basis for linking two people
            w = ds.club_idf.get(cid, 1.0) * cfg.SHARED_KIND_WEIGHT["club"]
            entries.append((i, col(f"c:{cid}"), w))
        m = str(major_of.get(p, ""))
        if m:
            rarity = math.log(max(2.0, n_people / max(1, majors_count[m])))
            entries.append((i, col(f"m:{m}"),
                            rarity * cfg.SHARED_KIND_WEIGHT["major"]))

    S = np.zeros((n_people, max(1, len(cols))), dtype=np.float32)
    for i, j, w in entries:
        S[i, j] = math.sqrt(max(0.0, w))
    return S, list(cols)


def person_links(ctx: GraphContext,
                 k: int | None = None) -> list[Edge]:
    """Each person's `k` strongest shared-characteristic links to other people.

    Computed as one matrix product rather than by enumerating pairs: the hub
    activity alone would generate over a million pairs, while S @ S.T is a
    1,800 x 1,800 float32 at 13 MB.

    An edge here is not an inference, it is an observed fact re-expressed: these
    two people share a rare activity. So it renders in the observed tier, not the
    inferred one.
    """
    k = k or cfg.SHARED_LINKS_PER_PERSON
    ds = ctx.ds
    people = list(ds.app_ids) + list(ds.alum_ids)
    S, _ = _shared_matrix(ds, people)

    P = (S @ S.T).astype(np.float32)
    np.fill_diagonal(P, -np.inf)

    kk = int(min(k, P.shape[1] - 1))
    idx = np.argpartition(-P, kk - 1, axis=1)[:, :kk]

    # Union of everyone's top k. Degree comes out uneven, from 3 up to about 33,
    # and that is left alone: a person who shows up in many other people's top
    # three genuinely is unusually well connected, and nobody ends up isolated.
    kept: set[tuple[int, int]] = set()
    for i in range(len(people)):
        for j in idx[i][np.argsort(-P[i, idx[i]])]:
            j = int(j)
            if np.isfinite(P[i, j]) and P[i, j] > 0:
                kept.add((i, j) if i < j else (j, i))

    return [Edge(people[a], people[b], "SHARES_WITH") for a, b in sorted(kept)]


# ------------------------------------------- NR-5: the three that matter
def edge_importance(ctx: GraphContext, e: Edge) -> float:
    """How much this edge tells you about the person it belongs to.

    Rarity of the target times a structural weight for the kind. A shared rare
    activity is informative; a shared major that half the cluster holds is not.
    """
    ds = ctx.ds
    base = cfg.EDGE_KIND_WEIGHT.get(e.kind, 1.0)
    target = e.target
    if e.kind in ("PARTICIPATES_IN", "PLAYS_FOR"):
        rarity = ds.act_idf.get(target, 1.0)
    elif e.kind in ("MEMBER_OF", "INTERESTED_IN"):
        rarity = ds.club_idf.get(target, 1.0)
    elif e.kind == "STUDIES":
        # how unusual this major is across both populations
        deg = len(ctx.adjacency.get(target, ()))
        rarity = math.log(max(2.0, (ds.n_app + ds.n_alum) / max(1, deg)))
    else:
        rarity = 1.0
    person = e.source if e.source in ds.app_index or e.source in ds.alum_index \
        else e.target
    role = max(ctx.ds.act_weights.get(person, {}).get(target, 0.0),
               ctx.ds.club_weights.get(person, {}).get(target, 0.0), 1.0)
    return base * rarity * role


def important_edges(ctx: GraphContext, k: int | None = None,
                    exclude_kinds: set[str] | None = None) -> list[Edge]:
    """At most `k` edges per person, the most informative ones.

    Drawing every edge gave some nodes a visual degree of eight and others one,
    which reads as noise rather than structure. A uniform degree is what makes
    the graph look homogeneous, and picking by importance means the edges that
    survive are the ones worth seeing.

    `exclude_kinds` drops edges the layout already encodes: when discs are
    majors, a STUDIES edge restates the node's own position.
    """
    k = k or cfg.TOP_EDGES_PER_PERSON
    exclude = exclude_kinds or set()
    ds = ctx.ds
    by_person: dict[str, list[tuple[float, Edge]]] = {}
    for e in ctx.edges:
        if e.kind in exclude:
            continue
        person = e.source if (e.source in ds.app_index
                              or e.source in ds.alum_index) else e.target
        by_person.setdefault(person, []).append((edge_importance(ctx, e), e))

    kept: dict[tuple[str, str, str], Edge] = {}
    for person, scored in by_person.items():
        # deterministic: importance descending, then target id
        scored.sort(key=lambda sv: (-sv[0], sv[1].target))
        for _, e in scored[:k]:
            kept[(e.source, e.target, e.kind)] = e
    return list(kept.values())


def cluster_layout(nodes: dict[str, Node], ds: Dataset, assignment
                   ) -> tuple[dict[str, tuple[float, float]], dict[str, dict]]:
    """Arrangement A, cohort clusters. The NR-4 default.

    Attribute nodes stay in the central region: they are shared infrastructure,
    not members of any cohort.
    """
    layout: dict[str, tuple[float, float]] = {}
    geo = cluster_geometry(assignment)

    for cohort, g in geo.items():
        cx, cy = g["centre"]
        ids = assignment.members[cohort]
        apps = sorted(i for i in ids if i in ds.app_index)
        alums = sorted(i for i in ids if i in ds.alum_index)
        # opposite offsets along the radial direction, so within a cluster the
        # two populations occupy distinguishable lobes (NR-4c)
        ox, oy = math.cos(g["angle"]), math.sin(g["angle"])
        for group, sign in ((apps, -1.0), (alums, 1.0)):
            lobe = g["radius"] * 0.74
            dx = ox * cfg.POPULATION_OFFSET * g["radius"] * sign
            dy = oy * cfg.POPULATION_OFFSET * g["radius"] * sign
            for nid, (px, py) in zip(group, _phyllotaxis(len(group), lobe)):
                layout[nid] = (cx + dx + px, cy + dy + py)

    # attribute nodes in the middle, laid out in compact bands by kind
    inner = {"Major": 0.55, "Activity": 0.20,
             "Athletic team": -0.20, "Campus club": -0.58}
    zig = 0.05
    y_max = max(abs(y) for y in inner.values()) + zig
    # derive the half-width so the band CORNER lands inside the clearing, not
    # just the band centre. A flat factor put the corners at radius 0.97 against
    # a declared clearing of 0.92.
    half = math.sqrt(max(0.01, cfg.CLUSTER_CENTRE_CLEAR ** 2 - y_max ** 2))
    for kind, y in inner.items():
        ids = sorted((n.id for n in nodes.values() if n.kind == kind),
                     key=lambda i: (nodes[i].meta.get("category", ""),
                                    nodes[i].label))
        m = max(1, len(ids) - 1)
        for i, nid in enumerate(ids):
            x = -half + 2 * half * (i / m)
            layout[nid] = (x, y + (zig if i % 2 else -zig))

    missing = set(nodes) - set(layout)
    if missing:
        raise AssertionError(f"{len(missing)} nodes have no position: "
                             f"{sorted(missing)[:5]}")
    return layout, geo


def fingerprint(ds: Dataset) -> str:
    h = hashlib.sha256()
    for part in (ds.n_app, ds.n_alum, len(ds.activities), len(ds.clubs),
                 len(ds.act_edges), len(ds.club_edges)):
        h.update(str(part).encode())
    return h.hexdigest()[:16]


def build_context(ds: Dataset, clustered: bool = True,
                  grouping: str = cfg.GROUP_BY_DEFAULT) -> GraphContext:
    """Every grouping's layout is precomputed and frozen.

    Switching grouping must not trigger a recompute, and no grouping depends on a
    control value, so NFR-6 holds: positions are a function of loaded data alone.
    """
    from core.cohorts import assign_grouping

    nodes = build_nodes(ds)
    edges = build_observed_edges(ds)
    adjacency = build_adjacency(edges)

    assignments: dict[str, object] = {}
    layouts: dict[str, dict] = {"arcs": freeze_layout(nodes, ds)}
    geometries: dict[str, dict] = {}
    for name in cfg.GROUPINGS:
        a = assign_grouping(ds, name)
        assignments[name] = a
        layouts[name], geometries[name] = cluster_layout(nodes, ds, a)

    ctx = GraphContext(
        ds=ds, nodes=nodes, edges=edges, adjacency=adjacency,
        layout=layouts[grouping if clustered else "arcs"],
        assignment=assignments[grouping],
        cluster_geo=geometries[grouping],
        layouts=layouts, clustered=clustered,
        assignments=assignments, geometries=geometries, grouping=grouping)
    ctx.apply_grouping(grouping, clustered)
    return ctx


# ------------------------------------------------------------ subgraphs
def similar_edges(ctx: GraphContext) -> list[Edge]:
    """NR-3a and NR-3b, materialised from the top-k selection."""
    if ctx.topk_idx is None:
        return []
    out: list[Edge] = []
    for i, row in enumerate(ctx.topk_idx):
        for e in row:
            if e >= 0:
                out.append(Edge(ctx.ds.app_ids[i], ctx.ds.alum_ids[int(e)],
                                "SIMILAR_TO"))
    return out


def drawable(ctx: GraphContext, ids: set[str]) -> set[str]:
    """Keep only what the graph draws: people and the activities linking them.

    Major, club and athletic-team ids stay in adjacency so shared-path
    highlighting and the club panels can use them, but they are not rendered.
    """
    allowed = cfg.PERSON_KINDS | cfg.CONNECTIVE_KINDS
    return {i for i in ids if ctx.nodes[i].kind in allowed}


def _induced(ctx: GraphContext, keep: set[str],
             include_similar: bool = True,
             only_drawable: bool = True) -> tuple[set[str], list[Edge]]:
    """Edges within a node set.

    When only people are drawable, observed connections come from the
    person-to-person projection rather than from the raw person-to-attribute
    list: those endpoints are characteristics now and are not on screen to
    connect to.
    """
    if only_drawable:
        keep = drawable(ctx, keep)
        edges = [e for e in ctx.person_links()
                 if e.source in keep and e.target in keep]
    else:
        edges = [e for e in ctx.edges
                 if e.source in keep and e.target in keep]
    if include_similar:
        edges += [e for e in similar_edges(ctx)
                  if e.source in keep and e.target in keep]
    return keep, edges


def _neighbourhood(ctx: GraphContext, seed: str, hops: int = 1) -> set[str]:
    frontier, seen = {seed}, {seed}
    for _ in range(hops):
        nxt: set[str] = set()
        for n in frontier:
            nxt |= ctx.adjacency.get(n, set())
        frontier = nxt - seen
        seen |= nxt
    return seen


def subgraph(mode: str, focus: str | None, ctx: GraphContext
             ) -> tuple[set[str], list[Edge]]:
    """NR-3d focus modes. Default is never the full population (NR-3c)."""
    ds = ctx.ds

    if mode == "Applicant focus" and focus:
        keep = {focus}
        keep |= ctx.adjacency.get(focus, set())
        # people who share characteristics with them: adjacency alone points at
        # the characteristics themselves, which are no longer drawn
        for e in ctx.person_links():
            if e.source == focus:
                keep.add(e.target)
            elif e.target == focus:
                keep.add(e.source)
        i = ds.app_index.get(focus)
        if i is not None and ctx.topk_idx is not None:
            for e in ctx.topk_idx[i]:
                if e >= 0:
                    alum = ds.alum_ids[int(e)]
                    keep.add(alum)
                    keep |= ctx.adjacency.get(alum, set())
        return _induced(ctx, keep)

    if mode == "Club focus" and focus:
        # clubs ARE the subject here, so they are drawn as nodes (AC-22 to AC-24)
        return _induced(ctx, _neighbourhood(ctx, focus), only_drawable=False)

    if mode == "Clubs at succession risk":
        from core.clubs import project_rosters
        ros = ctx.rosters if ctx.rosters is not None else project_rosters(ds)
        risky = [r.club_id for r in ros.itertuples(index=False) if r.at_risk]
        keep: set[str] = set()
        for cid in risky:
            keep |= _neighbourhood(ctx, cid)
        return _induced(ctx, keep, include_similar=False, only_drawable=False)

    if mode == "Cohort isolate" and focus:
        # IR-8c: one cohort, both populations, plus the attribute nodes those
        # people actually touch so the cluster is not floating in isolation
        keep = set(ctx.assignment.members.get(focus, [])) if ctx.assignment else set()
        for pid in list(keep):
            keep |= ctx.adjacency.get(pid, set())
        return _induced(ctx, keep)

    if mode == "Cohort overview":
        # People only. Every characteristic is a keyword on selection rather
        # than a node, so connections run person to person and no node carries a
        # degree of 170 because everyone attached to it.
        keep = {n for n, node in ctx.nodes.items()
                if node.kind in cfg.PERSON_KINDS}
        return keep, ctx.person_links()

    if mode == "Full population":
        return set(ctx.nodes), ctx.edges + similar_edges(ctx)

    # default: Short list plus the evidence behind each pick (IR-3b, IR-3c).
    # "Evidence" means the attribute nodes the applicant SHARES with a matched
    # alum, not every node they happen to touch. Including unshared attributes
    # would put nodes on screen that justify nothing, and at 285 picks it pulls
    # in almost the whole central band.
    keep: set[str] = set()
    if ctx.shortlist is not None:
        chosen = [ds.app_ids[i] for i in np.flatnonzero(ctx.shortlist)]
        keep |= set(chosen)
        for pid in chosen:
            i = ds.app_index[pid]
            if ctx.topk_idx is None:
                keep |= ctx.adjacency.get(pid, set())
                continue
            for e in ctx.topk_idx[i]:
                if e < 0:
                    continue
                alum = ds.alum_ids[int(e)]
                keep.add(alum)
                keep |= shared_nodes(ctx, pid, alum)
    return _induced(ctx, keep)


def shared_nodes(ctx: GraphContext, applicant: str, alum: str) -> set[str]:
    """XR-6: the observed nodes the pair actually share. This is the payoff for
    using a graph rather than asserting similarity in a tooltip."""
    return ctx.adjacency.get(applicant, set()) & ctx.adjacency.get(alum, set())


if __name__ == "__main__":
    import time
    from collections import Counter

    from core.clubs import project_rosters
    from core.features import build_tensors, flatten
    from core.loader import load
    from core.priorities import (blend_individual, build_shortlist,
                                 composition_matrix, p1_yield, p2_expected_ntr,
                                 p4_academic, percentile)
    from core.clubs import coverage_matrix
    from core.similarity import contract, topk, weights_from_words

    ds = load()
    t = time.perf_counter()
    ctx = build_context(ds)
    print(f"context built in {(time.perf_counter() - t) * 1000:.0f} ms")

    kinds = Counter(n.kind for n in ctx.nodes.values())
    print(f"\nnodes {len(ctx.nodes)} (specs §0.2 expects 1,925)")
    for k, v in kinds.items():
        print(f"  {k:16s} {v:5d}")
    tiers = Counter(e.tier for e in ctx.edges)
    print(f"observed edges {len(ctx.edges)}")
    for k, v in Counter(e.kind for e in ctx.edges).items():
        print(f"  {k:16s} {v:5d}")
    print(f"tiers {dict(tiers)}")

    # determinism of both frozen arrangements (NFR-2, NFR-6)
    a1, _ = cluster_layout(ctx.nodes, ds, ctx.assignment)
    a2, _ = cluster_layout(ctx.nodes, ds, ctx.assignment)
    print(f"\nclustered layout deterministic: {a1 == a2}")
    print(f"arc layout deterministic:       "
          f"{freeze_layout(ctx.nodes, ds) == freeze_layout(ctx.nodes, ds)}")
    print(f"arrangements cached: {sorted(ctx.layouts)}")

    print(f"\ncohort clusters on the ring (FR-8: a ring has no rank order)")
    print(f"{'#':>3} {'cohort':34s} {'n':>5} {'radius':>7}  centre")
    for cohort, g in ctx.cluster_geo.items():
        cx, cy = g["centre"]
        print(f"{g['index'] + 1:3d} {cohort:34s} {g['count']:5d} "
              f"{g['radius']:7.3f}  ({cx:+.2f}, {cy:+.2f})")

    # clusters must not overlap, or they stop reading as distinct groups
    import itertools
    worst = None
    for (ca, ga), (cb, gb) in itertools.combinations(ctx.cluster_geo.items(), 2):
        d = math.dist(ga["centre"], gb["centre"])
        gap = d - (ga["radius"] + gb["radius"])
        if worst is None or gap < worst[0]:
            worst = (gap, ca, cb)
    print(f"\ntightest cluster gap: {worst[0]:+.3f} between "
          f"{worst[1]} and {worst[2]}  "
          f"{'(overlapping)' if worst[0] < 0 else '(clear)'}")

    # attribute nodes must stay inside the central clearing
    attr = [n.id for n in ctx.nodes.values()
            if n.kind in ("Activity", "Athletic team", "Major", "Campus club")]
    far = max(math.hypot(*ctx.layout[i]) for i in attr)
    print(f"attribute nodes max radius: {far:.2f} "
          f"(clearing {cfg.CLUSTER_CENTRE_CLEAR})")

    # switching arrangement must be free and must actually move things
    before = dict(ctx.layout)
    ctx.set_arrangement(False)
    moved = sum(1 for k in before if before[k] != ctx.layout[k])
    print(f"IR-8d switch to arcs moved {moved} of {len(before)} nodes")
    ctx.set_arrangement(True)
    print(f"switch back restores clustered: {ctx.layout == before}")

    M, A = build_tensors(ds)
    M2, A2 = flatten(M, A)
    sim = contract(M2, A2, weights_from_words(cfg.DIM_DEFAULTS),
                   (ds.n_app, ds.n_alum))
    ctx.sim = sim
    ctx.topk_idx, ctx.topk_val = topk(sim, cfg.TOPK_EDGES, cfg.SIM_FLOOR)
    ctx.rosters = project_rosters(ds)

    p1, p2 = percentile(p1_yield(ds)), percentile(p2_expected_ntr(ds))
    p4 = p4_academic(sim, ds)[0]
    C, target, _ = composition_matrix(ds)
    B, _ = coverage_matrix(ds, ctx.rosters)
    id_rank = np.argsort(np.argsort(np.asarray(ds.app_ids)))
    ctx.shortlist = build_shortlist(
        blend_individual(p1, p2, p4, 0.5, 0.5, 0.5),
        C, target, B, None, 0.5, 0.25, cfg.CLASS_TARGET, id_rank)

    print("\nsubgraph sizes by focus mode (NR-3c: default is not full)")
    for mode in cfg.FOCUS_MODES:
        focus = None
        if mode == "Applicant focus":
            focus = ds.app_ids[0]
        elif mode == "Activity focus":
            focus = "ACT-01"
        elif mode == "Major cohort":
            focus = "MAJ::Biology"
        elif mode == "Club focus":
            focus = "CLB-01"
        elif mode == "Cohort isolate":
            focus = ctx.assignment.order[0]
        t = time.perf_counter()
        keep, edges = subgraph(mode, focus, ctx)
        ms = (time.perf_counter() - t) * 1000
        print(f"  {mode:28s} {len(keep):5d} nodes {len(edges):6d} edges "
              f"{ms:6.1f} ms")

    a, e = ds.app_ids[0], ds.alum_ids[int(ctx.topk_idx[0, 0])]
    sh = shared_nodes(ctx, a, e)
    print(f"\nXR-6 shared nodes for {a} / {e}: "
          f"{[ctx.nodes[s].label for s in sorted(sh)]}")
