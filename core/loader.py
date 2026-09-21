"""Load and validate every data contract. No Streamlit imports (specs §5).

Protected attributes are removed here rather than merely left unused downstream
(SR-3, FR-5). A column that never enters `Dataset` cannot leak into a score by
later carelessness, and the FR-5 audit becomes one assertion over `Dataset`
instead of an inspection of every code path.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402


class DataContractError(AssertionError):
    """Raised when a requirement-level data contract is violated."""


@dataclass
class Dataset:
    applicants: pd.DataFrame
    alumni: pd.DataFrame
    activities: pd.DataFrame
    act_edges: pd.DataFrame
    clubs: pd.DataFrame
    club_edges: pd.DataFrame

    # tensor row/column order is fixed here and nowhere else
    app_ids: list[str] = field(default_factory=list)
    alum_ids: list[str] = field(default_factory=list)
    app_index: dict[str, int] = field(default_factory=dict)
    alum_index: dict[str, int] = field(default_factory=dict)

    # adjacency, built once at load
    act_sets: dict[str, set[str]] = field(default_factory=dict)
    act_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    club_sets: dict[str, set[str]] = field(default_factory=dict)
    club_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    act_idf: dict[str, float] = field(default_factory=dict)
    club_idf: dict[str, float] = field(default_factory=dict)
    proxy_clubs: set[str] = field(default_factory=set)

    @property
    def n_app(self) -> int:
        return len(self.app_ids)

    @property
    def n_alum(self) -> int:
        return len(self.alum_ids)

    def activity_name(self, aid: str) -> str:
        row = self.activities.loc[self.activities.activity_id == aid, "activity_name"]
        return row.iloc[0] if len(row) else aid

    def club_name(self, cid: str) -> str:
        row = self.clubs.loc[self.clubs.club_id == cid, "club_name"]
        return row.iloc[0] if len(row) else cid


def _idf(sets: dict[str, set[str]], universe: list[str]) -> dict[str, float]:
    """SR-4: inverse document frequency over people, not over edges."""
    n = max(1, len(sets))
    df = {u: 0 for u in universe}
    for s in sets.values():
        for item in s:
            if item in df:
                df[item] += 1
    # +1 floor so a never-chosen item does not produce an infinite weight
    return {u: math.log(n / max(1, c)) if c else math.log(n) for u, c in df.items()}


def load(data_dir: Path | None = None) -> Dataset:
    d = Path(data_dir) if data_dir else cfg.DATA_DIR

    applicants = pd.read_csv(cfg.POOL_PATH)
    # SR-3 / FR-5: drop at the boundary.
    applicants = applicants.drop(columns=[c for c in cfg.PROTECTED_COLUMNS
                                          if c in applicants.columns])

    alumni = pd.read_csv(d / "alumni_outcomes.csv")
    activities = pd.read_csv(d / "activities_dim.csv")
    clubs = pd.read_csv(d / "campus_clubs_dim.csv")
    act_edges = pd.read_csv(d / "activities_edges.csv", keep_default_na=False)
    club_edges = pd.read_csv(d / "club_edges.csv", keep_default_na=False)

    app_ids = applicants.applicant_id.tolist()
    alum_ids = alumni.alum_id.tolist()

    ds = Dataset(
        applicants=applicants, alumni=alumni, activities=activities,
        act_edges=act_edges, clubs=clubs, club_edges=club_edges,
        app_ids=app_ids, alum_ids=alum_ids,
        app_index={a: i for i, a in enumerate(app_ids)},
        alum_index={a: i for i, a in enumerate(alum_ids)},
        proxy_clubs=set(clubs.loc[clubs.proxy_sensitive == "Y", "club_id"]),
    )

    # ---- adjacency with DR-2 role and recognition weighting
    for r in act_edges.itertuples(index=False):
        ds.act_sets.setdefault(r.person_id, set()).add(r.activity_id)
        w = (cfg.ROLE_MULT.get(r.role, 1.0)
             * cfg.RECOG_MULT.get(str(r.recognition_level), 1.0))
        slot = ds.act_weights.setdefault(r.person_id, {})
        slot[r.activity_id] = max(slot.get(r.activity_id, 0.0), w)

    # ---- club adjacency; applicants weighted by signal, alumni by role
    for r in club_edges.itertuples(index=False):
        ds.club_sets.setdefault(r.person_id, set()).add(r.club_id)
        if r.relationship == "INTERESTED_IN":
            w = cfg.SIGNAL_MULT.get(str(r.signal_strength), 1.0)
        else:
            w = cfg.CLUB_ROLE_MULT.get(str(r.club_role), 1.0)
        slot = ds.club_weights.setdefault(r.person_id, {})
        slot[r.club_id] = max(slot.get(r.club_id, 0.0), w)

    ds.act_idf = _idf(ds.act_sets, activities.activity_id.tolist())
    ds.club_idf = _idf(ds.club_sets, clubs.club_id.tolist())

    validate(ds)
    return ds


def validate(ds: Dataset) -> None:
    """The five checks from design.md 3.1. Fail loud, never coerce."""
    problems: list[str] = []

    # 1 ---- referential integrity, DR-2 and DR-6
    act_ok = set(ds.activities.activity_id)
    club_ok = set(ds.clubs.club_id)
    people = set(ds.app_ids) | set(ds.alum_ids)

    bad = set(ds.act_edges.activity_id) - act_ok
    if bad:
        problems.append(f"DR-2 unknown activity_id: {sorted(bad)[:5]}")
    bad = set(ds.act_edges.person_id) - people
    if bad:
        problems.append(f"DR-2 unknown person_id: {sorted(bad)[:5]}")
    bad = set(ds.club_edges.club_id) - club_ok
    if bad:
        problems.append(f"DR-6 unknown club_id: {sorted(bad)[:5]}")
    bad = set(ds.club_edges.person_id) - people
    if bad:
        problems.append(f"DR-6 unknown person_id: {sorted(bad)[:5]}")
    linked = {x for x in ds.clubs.linked_activity_id.dropna() if x != ""}
    bad = linked - act_ok
    if bad:
        problems.append(f"DR-6 unknown linked_activity_id: {sorted(bad)[:5]}")

    # 2 ---- relationship matches person_type
    ce = ds.club_edges
    mism = ce[((ce.relationship == "INTERESTED_IN") & (ce.person_type != "Applicant"))
              | ((ce.relationship == "MEMBER_OF") & (ce.person_type != "Alum"))]
    if len(mism):
        problems.append(f"DR-6 relationship/person_type mismatch: {len(mism)} rows")

    # 3 ---- null discipline by relationship, DR-6a
    interested = ce[ce.relationship == "INTERESTED_IN"]
    member = ce[ce.relationship == "MEMBER_OF"]
    if (interested.club_role.astype(str).str.len() > 0).any():
        problems.append("DR-6a INTERESTED_IN rows carry club_role")
    if (interested.years_active.astype(str).str.len() > 0).any():
        problems.append("DR-6a INTERESTED_IN rows carry years_active")
    if (member.interest_signal.astype(str).str.len() > 0).any():
        problems.append("DR-6a MEMBER_OF rows carry interest_signal")
    if (member.signal_strength.astype(str).str.len() > 0).any():
        problems.append("DR-6a MEMBER_OF rows carry signal_strength")

    # 4 ---- DR-6c, at least one club genuinely at succession risk
    yield_by = dict(zip(ds.applicants.applicant_id,
                        ds.applicants.predicted_yield_prob))
    pipe: dict[str, float] = {c: 0.0 for c in club_ok}
    for r in interested.itertuples(index=False):
        pipe[r.club_id] = pipe.get(r.club_id, 0.0) + yield_by.get(r.person_id, 0.0)
    at_risk = [
        r.club_id for r in ds.clubs.itertuples(index=False)
        if (r.current_members - r.graduating_members
            + pipe.get(r.club_id, 0.0)) < r.min_viable_members
    ]
    if not at_risk:
        problems.append("DR-6c no club is at succession risk; the club layer "
                        "has nothing to demonstrate")

    # 5 ---- protected attributes absent, SR-3 / FR-5
    for frame_name in ("applicants", "alumni"):
        cols = set(getattr(ds, frame_name).columns)
        leaked = cols & set(cfg.PROTECTED_COLUMNS)
        if leaked:
            problems.append(f"SR-3 protected column present in {frame_name}: "
                            f"{sorted(leaked)}")

    if problems:
        raise DataContractError("; ".join(problems))


def suppression_audit(ds: Dataset) -> dict[str, bool]:
    """FR-5: verifiable in the interface, not merely assumed."""
    present = set(ds.applicants.columns) | set(ds.alumni.columns)
    return {col: (col not in present) for col in cfg.PROTECTED_COLUMNS}


def at_risk_club_ids(ds: Dataset) -> list[str]:
    yield_by = dict(zip(ds.applicants.applicant_id,
                        ds.applicants.predicted_yield_prob))
    ce = ds.club_edges
    interested = ce[ce.relationship == "INTERESTED_IN"]
    pipe: dict[str, float] = {}
    for r in interested.itertuples(index=False):
        pipe[r.club_id] = pipe.get(r.club_id, 0.0) + yield_by.get(r.person_id, 0.0)
    return [
        r.club_id for r in ds.clubs.itertuples(index=False)
        if (r.current_members - r.graduating_members
            + pipe.get(r.club_id, 0.0)) < r.min_viable_members
    ]


if __name__ == "__main__":
    ds = load()
    print(f"applicants {ds.n_app}  alumni {ds.n_alum}  "
          f"activities {len(ds.activities)}  clubs {len(ds.clubs)}")
    print(f"activity edges {len(ds.act_edges)}  club edges {len(ds.club_edges)}")
    print(f"proxy_sensitive clubs {len(ds.proxy_clubs)}")
    print(f"at-risk clubs {len(at_risk_club_ids(ds))}")
    print(f"suppression audit {suppression_audit(ds)}")
    top = sorted(ds.act_idf.items(), key=lambda kv: kv[1])[:3]
    rare = sorted(ds.act_idf.items(), key=lambda kv: -kv[1])[:3]
    print("most common activities", [(ds.activity_name(a), round(v, 2)) for a, v in top])
    print("rarest activities     ", [(ds.activity_name(a), round(v, 2)) for a, v in rare])
    print("validation passed")
