"""Named rule-based cohort assignment (NR-4).

Both populations are assigned by the same predicates: applicants on their current
record, alumni on their `entry_*` block (NR-4a). That is the point of clustering.
An athlete cluster of applicants sitting beside the athlete cluster of alumni who
already graduated turns "did people like this thrive here" into a visual question.

FR-7 is enforced structurally. `Profile` carries no ethnicity or sex, because
`Dataset` carries none — they are dropped at load under SR-3. A predicate cannot
reference what it cannot see. `Profile.activities` holds activity ids only and
never club membership, so `proxy_sensitive` clubs cannot leak in either.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg               # noqa: E402
from core.loader import Dataset    # noqa: E402

GENERAL = "General cohort"
LOW_INCOME_BANDS = {"Under $30K", "$30K - $59K"}

# activity ids referenced by the conjunction-defined cohorts
CIVIC_ACTS = {"ACT-01", "ACT-02", "ACT-17", "ACT-18"}   # debate, MUN, stugov
RESEARCH_ACTS = {"ACT-27", "ACT-28", "ACT-29"}          # research, published, lab
ARTS_ACTS = {"ACT-07", "ACT-08", "ACT-09", "ACT-10", "ACT-11", "ACT-12"}

CIVIC_MAJORS = {"Political Science", "History", "English"}
RESEARCH_MAJORS = {"Biology", "Chemistry", "Neuroscience",
                   "Environmental Science", "Mathematics"}


def _s(value) -> str:
    """Empty string for a missing value.

    `value or ""` does NOT do this: numpy NaN is truthy, so it survives the `or`
    and `str()` turns it into the literal "nan", which is a non-empty string.
    That single trap put all 1,800 people in Recruited athletes on the first run.
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>") else text


@dataclass(frozen=True)
class Profile:
    """One shape so a single predicate can serve both schemas."""
    pid: str
    population: str
    sport: str
    school: str
    major: str
    rating: int
    gpa: float
    first_gen: bool
    income: str
    international: bool
    interest: int
    activities: frozenset          # activity_id only, never clubs
    recognised_arts: bool          # arts activity at State or National level


@dataclass(frozen=True)
class CohortDef:
    order: int
    name: str
    description: str
    predicate: Callable[[Profile], bool]


# ------------------------------------------------------------ NR-4f cohorts
# `order` is the DECLARATION order, used only as a deterministic tie-break.
# Actual precedence is computed by ascending predicate breadth: see `precedence`.
COHORT_DEFS: list[CohortDef] = [
    CohortDef(
        1, "Recruited athletes",
        "A varsity sport is present: the recruited sport for applicants, a "
        "season played for alumni.",
        lambda p: bool(p.sport)),
    CohortDef(
        2, "Performing and visual arts",
        "College of Fine Arts, or an arts activity carrying State or National "
        "recognition.",
        lambda p: p.school == "College of Fine Arts" or p.recognised_arts),
    CohortDef(
        3, "Pre-law and civic",
        "Political Science, History or English, AND one of Debate and Speech, "
        "Model UN, Student Government or Class Officer. A conjunction: neither "
        "half alone qualifies.",
        lambda p: p.major in CIVIC_MAJORS and bool(p.activities & CIVIC_ACTS)),
    CohortDef(
        4, "Research-track sciences",
        "Biology, Chemistry, Neuroscience, Environmental Science or "
        "Mathematics, AND one of Independent Research, Published Work or Lab "
        "Assistantship. Also a conjunction.",
        lambda p: p.major in RESEARCH_MAJORS
        and bool(p.activities & RESEARCH_ACTS)),
    CohortDef(
        5, "Engineering and applied science",
        "School of Engineering: the intended school for applicants, the school "
        "at entry for alumni (DR-4a).",
        lambda p: p.school == "School of Engineering"),
    CohortDef(
        6, "Health sciences",
        "School of Nursing and Health: intended school for applicants, school "
        "at entry for alumni (DR-4a).",
        lambda p: p.school == "School of Nursing & Health"),
    CohortDef(
        7, "Business and enterprise",
        "School of Business: intended school for applicants, school at entry "
        "for alumni (DR-4a).",
        lambda p: p.school == "School of Business"),
    CohortDef(
        8, "Top academic",
        "Reader rating of 5, or recalculated GPA at or above 3.90. Entering "
        "credentials only; it says nothing about how anyone did here.",
        lambda p: p.rating >= 5 or p.gpa >= 3.90),
    CohortDef(
        9, "International",
        "International citizenship for applicants, international region of "
        "origin for alumni.",
        lambda p: p.international),
    CohortDef(
        10, "Access and first-generation",
        "First-generation AND a family income band below $60K. Visible and "
        "removable: a cluster is a stronger act than a scoring weight (FR-7).",
        lambda p: p.first_gen and p.income in LOW_INCOME_BANDS),
    CohortDef(
        11, "Highly engaged",
        "Demonstrated interest rating of 4 or 5.",
        lambda p: p.interest >= 4),
    CohortDef(
        12, GENERAL,
        "Everything else, plus any cohort merged for being too small to read "
        "(NR-4e).",
        lambda p: True),
]

COHORT_NAMES = [d.name for d in COHORT_DEFS]
MIN_COHORT_SIZE = 40          # NR-4e; below this a cluster reads as noise


@dataclass
class Assignment:
    primary: dict[str, str]                 # person_id -> cohort name
    memberships: dict[str, list[str]]       # person_id -> every cohort matched
    members: dict[str, list[str]]           # cohort -> person ids
    merged: list[str]                       # cohorts folded into General
    order: list[str]                        # computed precedence, for XR-12

    def counts(self) -> dict[str, int]:
        return {c: len(v) for c, v in self.members.items()}

    def split(self, ds: Dataset, cohort: str) -> tuple[list[str], list[str]]:
        ids = self.members.get(cohort, [])
        return ([p for p in ids if p in ds.app_index],
                [p for p in ids if p in ds.alum_index])


def definitions() -> list[CohortDef]:
    """XR-12 reads from the same objects the predicates use, so the disclosure
    cannot drift from the behaviour."""
    return list(COHORT_DEFS)


# ------------------------------------------------------------- profiles
def _recognised_arts(ds: Dataset, pid: str) -> bool:
    rows = ds.act_edges[ds.act_edges.person_id == pid]
    for r in rows.itertuples(index=False):
        if r.activity_id in ARTS_ACTS and str(r.recognition_level) in (
                "State", "National"):
            return True
    return False


def build_profiles(ds: Dataset) -> list[Profile]:
    strong_arts: dict[str, bool] = {}
    for r in ds.act_edges.itertuples(index=False):
        if r.activity_id in ARTS_ACTS and str(r.recognition_level) in (
                "State", "National"):
            strong_arts[r.person_id] = True

    out: list[Profile] = []
    for r in ds.applicants.itertuples(index=False):
        pid = r.applicant_id
        out.append(Profile(
            pid=pid, population="Applicant",
            sport=_s(r.athletic_recruit_sport),
            school=_s(r.intended_school), major=_s(r.intended_major),
            rating=int(r.academic_rating), gpa=float(r.hs_gpa_recalc),
            first_gen=_s(r.first_generation) == "Y",
            income=_s(r.family_income_band),
            international=_s(r.citizenship) == "International",
            interest=int(r.demonstrated_interest_rating),
            activities=frozenset(ds.act_sets.get(pid, set())),
            recognised_arts=strong_arts.get(pid, False)))

    for r in ds.alumni.itertuples(index=False):
        pid = r.alum_id
        out.append(Profile(
            pid=pid, population="Alum",
            sport=_s(r.entry_athletic_recruit_sport),
            school=_s(r.entry_intended_school),
            major=_s(r.entry_intended_major),
            rating=int(r.entry_academic_rating),
            gpa=float(r.entry_hs_gpa_recalc),
            first_gen=_s(r.entry_first_generation) == "Y",
            income=_s(r.entry_family_income_band),
            international=_s(r.entry_region) == "International",
            interest=int(r.entry_demonstrated_interest_rating),
            activities=frozenset(ds.act_sets.get(pid, set())),
            recognised_arts=strong_arts.get(pid, False)))
    return out


# ------------------------------------------------------------ assignment
def precedence(profiles: list[Profile]) -> list[str]:
    """Precedence by ascending predicate breadth, rarest first.

    The first attempt hand-ordered this as "smaller and operationally distinct
    first" and it failed on contact with the data: 64 people matched Access and
    first-generation, but only 10 reached it, because broad school-track
    predicates upstream claimed the rest. The cohort then fell below the NR-4e
    floor and disappeared, breaking FR-7's requirement that it ship visible.

    Ordering by rarity fixes that without anyone deciding which group matters
    more. It is a legibility rule, not a policy judgement: the narrowest
    predicates get first claim, so the most distinctive groups stay large enough
    to read. It is also self-maintaining if the data changes.

    Deterministic for a given dataset, which is what NFR-6 requires; precedence
    depends only on loaded data, never on a control value.
    """
    breadth = []
    for d in COHORT_DEFS:
        if d.name == GENERAL:
            continue
        n = sum(1 for p in profiles if d.predicate(p))
        breadth.append((n, d.order, d.name))
    return [name for _, _, name in sorted(breadth)] + [GENERAL]


def assign(ds: Dataset, min_size: int = MIN_COHORT_SIZE) -> Assignment:
    profiles = build_profiles(ds)
    order = precedence(profiles)
    by_name = {d.name: d for d in COHORT_DEFS}

    memberships: dict[str, list[str]] = {}
    primary: dict[str, str] = {}
    for p in profiles:
        # report memberships in precedence order so the first entry IS the
        # primary; a different order here would make the two disagree
        hits = [name for name in order if by_name[name].predicate(p)]
        memberships[p.pid] = hits
        primary[p.pid] = hits[0] if hits else GENERAL

    members: dict[str, list[str]] = {c: [] for c in order}
    for pid, cohort in primary.items():
        members[cohort].append(pid)

    # NR-4e: merge undersized cohorts, counted over both populations jointly.
    # A cohort with four applicants and no alumni is noise, and it also breaks
    # the NR-4a comparison the clustering exists to enable.
    merged: list[str] = []
    for cohort in order:
        if cohort == GENERAL:
            continue
        ids = members[cohort]
        has_app = any(p in ds.app_index for p in ids)
        has_alum = any(p in ds.alum_index for p in ids)
        if len(ids) < min_size or not (has_app and has_alum):
            merged.append(cohort)
            members[GENERAL].extend(ids)
            for p in ids:
                primary[p] = GENERAL
            members[cohort] = []

    members = {c: sorted(v) for c, v in members.items() if v}
    # keep memberships consistent with the post-merge primary
    for pid, hits in memberships.items():
        memberships[pid] = [h for h in hits if h not in merged] or [GENERAL]
    return Assignment(primary=primary, memberships=memberships,
                      members=members, merged=merged, order=order)


def cohort_of(assignment: Assignment, pid: str) -> str:
    return assignment.primary.get(pid, GENERAL)


# --------------------------------------------- alternative groupings
# Position can encode only one grouping, so these are alternatives to the
# cohort discs rather than additions to them.
OTHER_LABEL = "Other"


def assign_by_field(ds: Dataset, field: str,
                    min_size: int = MIN_COHORT_SIZE) -> Assignment:
    """Group by a single attribute, e.g. major or school.

    Reuses the cohort machinery so the renderer and panels need no special case:
    same Assignment shape, same NR-4e merge rule, same ordering discipline.
    Memberships are single-valued here by construction, because an attribute is.
    """
    profiles = build_profiles(ds)
    value_of = {"major": lambda p: p.major, "school": lambda p: p.school}[field]

    primary = {p.pid: (value_of(p) or OTHER_LABEL) for p in profiles}
    members: dict[str, list[str]] = {}
    for pid, key in primary.items():
        members.setdefault(key, []).append(pid)

    # NR-4e again: a disc with four people reads as noise, and a single-population
    # disc breaks the NR-4a comparison the clustering exists to enable
    merged: list[str] = []
    for key in list(members):
        if key == OTHER_LABEL:
            continue
        ids = members[key]
        has_app = any(p in ds.app_index for p in ids)
        has_alum = any(p in ds.alum_index for p in ids)
        if len(ids) < min_size or not (has_app and has_alum):
            merged.append(key)
            members.setdefault(OTHER_LABEL, []).extend(ids)
            for p in ids:
                primary[p] = OTHER_LABEL
            del members[key]

    # largest first, so the biggest groups get stable ring positions; Other last
    order = sorted((k for k in members if k != OTHER_LABEL),
                   key=lambda k: (-len(members[k]), k))
    if OTHER_LABEL in members:
        order.append(OTHER_LABEL)

    return Assignment(
        primary=primary,
        memberships={p: [primary[p]] for p in primary},
        members={k: sorted(v) for k, v in members.items()},
        merged=merged, order=order)


def assign_grouping(ds: Dataset, grouping: str) -> Assignment:
    if grouping == "Major":
        return assign_by_field(ds, "major")
    if grouping == "School":
        return assign_by_field(ds, "school")
    return assign(ds)


def grouping_definitions(grouping: str, assignment: Assignment) -> list[CohortDef]:
    """XR-12 applies to any grouping: a disc label is an assertion about people."""
    if grouping == "Cohort":
        return definitions()
    noun = "major" if grouping == "Major" else "school"
    out = []
    for i, key in enumerate(assignment.order, start=1):
        if key == OTHER_LABEL:
            desc = (f"Every {noun} too small to read as its own disc, merged "
                    f"under NR-4e: {', '.join(assignment.merged) or 'none'}.")
        else:
            desc = (f"Intended {noun} for applicants, {noun} at entry for "
                    f"alumni (DR-4a).")
        out.append(CohortDef(i, key, desc, lambda p: False))
    return out


if __name__ == "__main__":
    from core.loader import load

    ds = load()
    a = assign(ds)

    print(f"cohorts in use: {len(a.members)} of {len(COHORT_NAMES)}")
    print(f"merged into General (NR-4e): {a.merged or 'none'}")
    print(f"\ncomputed precedence, rarest predicate first (XR-12):")
    print(f"{'#':>3} {'cohort':34s} {'total':>6} {'app':>6} {'alum':>6}")
    for i, c in enumerate(a.order, start=1):
        if c not in a.members:
            continue
        app, alum = a.split(ds, c)
        print(f"{i:3d} {c:34s} {len(a.members[c]):6d} {len(app):6d} "
              f"{len(alum):6d}")

    total = sum(len(v) for v in a.members.values())
    print(f"\ntotal assigned {total} (expected {ds.n_app + ds.n_alum})")
    print(f"every person has a primary: "
          f"{len(a.primary) == ds.n_app + ds.n_alum}")

    multi = [p for p, m in a.memberships.items() if len(m) > 2]
    print(f"people matching 3+ cohorts: {len(multi)} "
          f"(position is single-valued, memberships are not)")

    # precedence demonstration: an athlete who is also top academic
    demo = [p for p in build_profiles(ds)
            if p.sport and (p.rating >= 5 or p.gpa >= 3.90)]
    if demo:
        p = demo[0]
        print(f"\nprecedence check: {p.pid} has a sport AND "
              f"rating {p.rating}/GPA {p.gpa}")
        print(f"  matches      {a.memberships[p.pid]}")
        print(f"  primary      {a.primary[p.pid]}")

    print("\nFR-7 structural check")
    fields = set(Profile.__dataclass_fields__)
    print(f"  Profile fields: {sorted(fields)}")
    print(f"  protected fields present: "
          f"{sorted(fields & set(cfg.PROTECTED_COLUMNS)) or 'none'}")
    club_ids = set(ds.clubs.club_id)
    leaked = [p.pid for p in build_profiles(ds)[:200] if p.activities & club_ids]
    print(f"  club ids leaked into Profile.activities: {len(leaked)}")
