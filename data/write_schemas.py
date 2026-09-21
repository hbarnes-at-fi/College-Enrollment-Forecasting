"""Emit a schema .md per data file, generated from the CSVs themselves.

Hand-transcribing 40-column schemas invites drift between the document and the
data. These are derived, so they cannot disagree.

Run: python -m data.write_schemas
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg           # noqa: E402
from data import vocab as V    # noqa: E402

MAX_ENUM = 12          # beyond this, treat as free text rather than an enum


def infer_type(values: list[str]) -> str:
    vals = [v for v in values if v != ""]
    nullable = len(vals) < len(values)
    if not vals:
        return "string, always empty"
    kind = "string"
    if all(v.lstrip("-").isdigit() for v in vals):
        kind = "integer"
    else:
        try:
            for v in vals:
                float(v)
            kind = "decimal"
        except ValueError:
            kind = "string"
    uniq = sorted(set(vals))
    if kind == "string" and len(uniq) <= MAX_ENUM:
        kind = "enum"
    if set(uniq) <= {"Y", "N"}:
        kind = "Y/N"
    return kind + (", nullable" if nullable else "")


def enum_values(values: list[str]) -> str | None:
    vals = sorted({v for v in values if v != ""})
    if 1 < len(vals) <= MAX_ENUM and not all(
            v.lstrip("-").replace(".", "", 1).isdigit() for v in vals):
        return ", ".join(vals)
    return None


def write_schema(path: Path, title: str, notes: list[str],
                 sample_rows: tuple[int, int] = (0, 1)) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
    a, b = (rows[i] for i in sample_rows)

    out = [f"# Schema: `{path.name}`", ""]
    out.append(f"{len(rows):,} rows x {len(cols)} columns. {title}")
    out.append("")
    for n in notes:
        out.append(n)
        out.append("")
    out.append("Generated from the CSV by `data/write_schemas.py`; the examples "
               "are real rows, not illustrations.")
    out.append("")
    out.append("| Column | Type | Example A | Example B |")
    out.append("| :--- | :--- | :--- | :--- |")
    for c in cols:
        series = [r[c] for r in rows]
        out.append(f"| `{c}` | {infer_type(series)} | {a[c]} | {b[c]} |")

    enums = []
    for c in cols:
        ev = enum_values([r[c] for r in rows])
        if ev:
            enums.append(f"- `{c}`: {ev}")
    if enums:
        out += ["", "## Enum values", ""] + enums

    md = path.with_suffix("").with_suffix(".schema.md")
    md.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  {md.name:36s} {len(rows):6,} rows  {len(cols):2d} cols")


def main() -> None:
    d = cfg.DATA_DIR
    print("writing schemas")

    write_schema(
        d / "activities_dim.csv",
        "The DR-3 controlled vocabulary. One shared namespace across both "
        "populations; without it nothing links.",
        [f"{len(V.NON_ATHLETIC)} non-athletic activities plus "
         f"{len(V.SPORTS)} sports. Sport names match `athletic_recruit_sport` "
         "in the admit pool exactly, so DR-1 athletics join without a mapping "
         "table. Sports are rendered as the Athletic team node type, not as "
         "Activity, per specs.md section 0.2 — one node per sport, not two."])

    write_schema(
        d / "campus_clubs_dim.csv",
        "The DR-6 campus club roster. Distinct from activities: these are "
        "organisations that exist here, each with a headcount and a viability "
        "floor.",
        ["`linked_activity_id` is the bridge that connects a high school "
         "activity to its campus counterpart, so HS Debate reaches the Debate "
         "Society. Seven clubs deliberately have none; a bridge that always "
         "exists carries no information.",
         "`proxy_sensitive = Y` marks affinity and religious organisations. "
         "These are included in succession projection, roster panels, and the "
         "graph, but **excluded from similarity scoring and from P3 coverage "
         "credit** (specs.md section 0.1). Interest in one is a proxy for race "
         "or religion, which SR-3 excludes.",
         "A club is at succession risk when "
         "`current_members - graduating_members + expected new members` falls "
         "below `min_viable_members`. Nine of 45 are, per DR-6c."])

    write_schema(
        d / "alumni_outcomes.csv",
        "The DR-4 outcome cohort. Two blocks, and the split is the point.",
        ["**Entry block** (`entry_*`) mirrors the admit pool field for field. "
         "Applicants are compared only against this block; scoring an applicant "
         "against an alum's final GPA is a category error, since the applicant "
         "has no counterpart to it (DR-4a).",
         "**Outcome block** carries the four success components separately "
         "(DR-4c). They are never collapsed into one hidden number, because a "
         "single blended score conceals which definition of success is doing "
         "the work, and the four definitions select different people.",
         "**No giving field.** Alumni giving is a wealth proxy and is not "
         "captured anywhere (DR-4b)."])

    write_schema(
        d / "activities_edges.csv",
        "The DR-2 many-to-many activity edge list, spanning both populations.",
        ["The admit pool carries exactly one `special_talent_tag` per "
         "applicant. An activity graph needs many-to-many, which is why "
         "activities live here rather than in a column.",
         "`role` and `recognition_level` weight the edge in the SR-4 "
         "rarity-weighted Jaccard. Shared membership in a hub activity is weak "
         "evidence; shared membership in a rare one is strong."])

    write_schema(
        d / "club_edges.csv",
        "The DR-6 club edge list. One file, two relationships, discriminated by "
        "`relationship`.",
        ["`INTERESTED_IN` (applicants) and `MEMBER_OF` (alumni) carry genuinely "
         "different attributes, so columns are nullable by relationship "
         "(DR-6a). The loader rejects any row that violates this.",
         "The two render differently: `MEMBER_OF` is observed past, drawn "
         "solid; `INTERESTED_IN` is observed *intent* about an unrealised "
         "future, drawn dashed. A portal click must not look like four years of "
         "membership (NR-2a).",
         "`signal_strength` is a word scale in the data, not a number, so the "
         "interface never has to render a club interest score (DR-6b)."],
        sample_rows=(0, -1))


if __name__ == "__main__":
    main()
