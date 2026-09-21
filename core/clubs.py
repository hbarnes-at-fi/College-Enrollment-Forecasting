"""Club affinity (SR-7) and succession projection (SR-8).

Two rules drive everything here:

SR-7b  Two people sharing no clubs are not similar. Where both sides are empty
       the dimension leaves the normaliser rather than scoring 0 or 1.
§0.1   `proxy_sensitive` clubs are projected, rendered and panelled, but never
       scored. Interest in an affinity or religious organisation is a proxy for
       race or religion, which SR-3 excludes. They are dropped from the
       similarity dimension and from P3 coverage credit, and kept everywhere
       else, because a dying chapter is real information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg               # noqa: E402
from core.loader import Dataset    # noqa: E402


def _bridged_weights(ds: Dataset) -> dict[str, dict[str, float]]:
    """SR-7a: stated interest backed by the linked high school activity counts
    for more than a portal click. Boost capped at 1.0."""
    linked = dict(zip(ds.clubs.club_id,
                      ds.clubs.linked_activity_id.fillna("").astype(str)))
    out: dict[str, dict[str, float]] = {}
    for pid, clubs in ds.club_sets.items():
        done = ds.act_sets.get(pid, set())
        base = ds.club_weights.get(pid, {})
        slot = {}
        for cid in clubs:
            w = base.get(cid, 1.0)
            act = linked.get(cid, "")
            if act and act in done:
                w = min(1.0 * cfg.BRIDGE_BOOST, w * cfg.BRIDGE_BOOST)
            slot[cid] = w
        out[pid] = slot
    return out


def club_affinity(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Dimension 8. Applicant INTERESTED_IN against alum MEMBER_OF."""
    from core.features import _tanimoto, _weighted_sets

    weights = _bridged_weights(ds)
    universe = ds.clubs.club_id.tolist()
    drop = set(ds.proxy_clubs)          # §0.1

    Sa, na = _weighted_sets(ds.app_ids, universe, ds.club_sets,
                            weights, ds.club_idf, drop=drop)
    Se, ne = _weighted_sets(ds.alum_ids, universe, ds.club_sets,
                            weights, ds.club_idf, drop=drop)
    return _tanimoto(Sa, na, Se, ne)


# ------------------------------------------------------------------- SR-8
def _band_for(projected: float, min_viable: int) -> str:
    ratio = projected / max(1, min_viable)
    for name, floor in cfg.ROSTER_BANDS:
        if ratio >= floor:
            return name
    return cfg.ROSTER_BANDS[-1][0]


def project_rosters(ds: Dataset) -> pd.DataFrame:
    """projected = current - graduating + expected new members.

    SR-8d: the sum is probability-weighted, so it is an expectation and never a
    headcount. Callers must display `band`, not `projected`.
    """
    yield_by = dict(zip(ds.applicants.applicant_id,
                        ds.applicants.predicted_yield_prob))
    ce = ds.club_edges
    interested = ce[ce.relationship == "INTERESTED_IN"]

    pipe: dict[str, float] = {}
    heads: dict[str, list[str]] = {}
    for r in interested.itertuples(index=False):
        pipe[r.club_id] = pipe.get(r.club_id, 0.0) + yield_by.get(r.person_id, 0.0)
        heads.setdefault(r.club_id, []).append(r.person_id)

    members: dict[str, list[str]] = {}
    for r in ce[ce.relationship == "MEMBER_OF"].itertuples(index=False):
        members.setdefault(r.club_id, []).append(r.person_id)

    rows = []
    for r in ds.clubs.itertuples(index=False):
        holdover = r.current_members - r.graduating_members
        expected = pipe.get(r.club_id, 0.0)
        projected = holdover + expected
        band = _band_for(projected, r.min_viable_members)
        rows.append({
            "club_id": r.club_id,
            "club_name": r.club_name,
            "category": r.category,
            "linked_activity_id": str(r.linked_activity_id or ""),
            "current_members": r.current_members,
            "graduating_members": r.graduating_members,
            "holdover": holdover,
            "min_viable_members": r.min_viable_members,
            "expected_new": round(expected, 2),
            "projected": round(projected, 2),
            "band": band,
            "at_risk": projected < r.min_viable_members,   # SR-8a
            "interested_count": len(heads.get(r.club_id, [])),
            "alumni_members": len(members.get(r.club_id, [])),
            "charter_status": r.charter_status,
            "advisor_backed": r.advisor_backed,
            "proxy_sensitive": r.proxy_sensitive == "Y",
        })
    return pd.DataFrame(rows)


def interested_applicants(ds: Dataset, club_id: str) -> list[str]:
    ce = ds.club_edges
    m = (ce.relationship == "INTERESTED_IN") & (ce.club_id == club_id)
    return ce.loc[m, "person_id"].tolist()


def member_alumni(ds: Dataset, club_id: str) -> list[str]:
    ce = ds.club_edges
    m = (ce.relationship == "MEMBER_OF") & (ce.club_id == club_id)
    return ce.loc[m, "person_id"].tolist()


def sole_pipeline(ds: Dataset, rosters: pd.DataFrame | None = None,
                  max_heads: int = 2) -> dict[str, list[str]]:
    """SR-8b. At-risk clubs whose entire realistic pipeline is one or two people.

    Zero interested is not a sole pipeline, it is no pipeline, and gives IR-7e
    nothing to surface.
    """
    rosters = rosters if rosters is not None else project_rosters(ds)
    out: dict[str, list[str]] = {}
    for r in rosters.itertuples(index=False):
        if not r.at_risk:
            continue
        heads = interested_applicants(ds, r.club_id)
        if 1 <= len(heads) <= max_heads:
            out[r.club_id] = heads
    return out


def sole_pipeline_applicants(ds: Dataset,
                             rosters: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Inverted view: applicant -> the at-risk clubs they alone would carry."""
    inv: dict[str, list[str]] = {}
    for cid, heads in sole_pipeline(ds, rosters).items():
        for pid in heads:
            inv.setdefault(pid, []).append(cid)
    return inv


def coverage_matrix(ds: Dataset,
                    rosters: pd.DataFrame | None = None
                    ) -> tuple[np.ndarray, list[str]]:
    """B for the P3 greedy loop (SR-8c).

    (n_app, n_at_risk) binary. `proxy_sensitive` clubs are excluded: coverage
    credit would reward admitting by inferred identity.
    """
    rosters = rosters if rosters is not None else project_rosters(ds)
    target = rosters[(rosters.at_risk) & (~rosters.proxy_sensitive)]
    club_ids = target.club_id.tolist()
    B = np.zeros((ds.n_app, len(club_ids)), dtype=np.float32)
    col = {c: j for j, c in enumerate(club_ids)}
    ce = ds.club_edges
    for r in ce[ce.relationship == "INTERESTED_IN"].itertuples(index=False):
        j = col.get(r.club_id)
        i = ds.app_index.get(r.person_id)
        if j is not None and i is not None:
            B[i, j] = 1.0
    return B, club_ids


if __name__ == "__main__":
    from core.loader import load

    ds = load()
    ros = project_rosters(ds)
    print(f"clubs {len(ros)}   at risk {int(ros.at_risk.sum())}   "
          f"proxy_sensitive {int(ros.proxy_sensitive.sum())}")
    print("\nband distribution")
    print(ros.band.value_counts().to_string())
    print("\nat-risk clubs")
    cols = ["club_id", "club_name", "holdover", "expected_new",
            "min_viable_members", "band", "interested_count", "proxy_sensitive"]
    print(ros[ros.at_risk][cols].to_string(index=False))
    sp = sole_pipeline(ds, ros)
    print(f"\nsole-pipeline clubs (SR-8b): {len(sp)}")
    for cid, heads in sp.items():
        print(f"  {cid} {ds.club_name(cid):28s} carried by {heads}")
    B, ids = coverage_matrix(ds, ros)
    print(f"\ncoverage matrix {B.shape}  clubs scored for P3: {ids}")
    print(f"proxy clubs excluded from P3 coverage: "
          f"{sorted(set(ros[ros.at_risk & ros.proxy_sensitive].club_id))}")
