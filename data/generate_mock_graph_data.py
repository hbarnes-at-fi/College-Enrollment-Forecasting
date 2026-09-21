"""Seeded generator for DR-2, DR-3, DR-4 and DR-6.

Writes activities_dim, campus_clubs_dim, alumni_outcomes, activities_edges and
club_edges. Deterministic under config.SEED (NFR-5).

Run: python -m data.generate_mock_graph_data   (from the project root)
"""
from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg           # noqa: E402
from data import vocab as V    # noqa: E402

rng = random.Random(cfg.SEED)

# DR-1 special_talent_tag -> DR-3 activity, so generated edges agree with the
# existing admit pool instead of contradicting it.
TALENT_TO_ACT = {
    "Music - Instrumental": "ACT-07",
    "Music - Vocal": "ACT-08",
    "Studio Art Portfolio": "ACT-11",
    "Theatre Audition": "ACT-09",
    "Dance Audition": "ACT-10",
    "Debate / Speech": "ACT-01",
    "Robotics / STEM Competition": "ACT-05",
    "Published Research": "ACT-28",
}

REGIONS = ["Mid-Atlantic", "New England", "Midwest", "South", "Southwest",
           "West", "International"]
REGION_W = [34, 11, 12, 15, 5, 10, 13]
HS_TYPES = ["Public", "Private - Independent", "Parochial", "Charter",
            "Home School", "International School"]
HS_W = [60, 16, 11, 6, 2, 5]
INCOME = ["Under $30K", "$30K - $59K", "$60K - $99K", "$100K - $149K",
          "$150K - $249K", "$250K and above", "Not Reported"]
INCOME_W = [12, 15, 18, 22, 20, 8, 5]
SCHOOL_MAJORS = {
    "College of Arts & Sciences": [
        "Biology", "Psychology", "Political Science", "English", "Economics",
        "Chemistry", "Neuroscience", "History", "Environmental Science",
        "Mathematics", "Communication", "Sociology", "Undeclared"],
    "School of Engineering": [
        "Mechanical Engineering", "Computer Science", "Electrical Engineering",
        "Civil Engineering", "Biomedical Engineering", "Chemical Engineering"],
    "School of Business": [
        "Finance", "Marketing", "Accounting", "Business Analytics",
        "Management", "Supply Chain Management"],
    "School of Nursing & Health": [
        "Nursing (BSN)", "Public Health", "Exercise Science",
        "Health Administration"],
    "College of Fine Arts": [
        "Music Performance", "Studio Art", "Theatre", "Graphic Design"],
}
SCHOOL_W = [42, 19, 20, 13, 6]
POSTGRAD = ["Employed full-time", "Graduate or professional school",
            "Fellowship", "Seeking", "Unknown"]
POSTGRAD_W = [52, 26, 5, 12, 5]

# Applicant club-interest count, mean 1.81, P(0)=0.14 (specs §4.2)
INTEREST_COUNT_W = [0.14, 0.28, 0.30, 0.19, 0.09]
# Alum club-membership count, mean 2.10 (specs §4.2)
MEMBER_COUNT_W = [0.10, 0.22, 0.30, 0.24, 0.14]

ENTRY_OUTCOME_LOAD = 0.42      # calibrated to Spearman ~0.35 (specs §4.3)
LINKED_INTEREST_PULL = 42.0    # calibrated to Cramer's V ~0.45 (DR-6d)
# DR-6c: a club is at risk BECAUSE few applicants want it. Suppressing interest
# here is what makes the flag coherent; forcing it in the roster arithmetic
# alone produced clubs with 39 interested applicants labelled "at risk".
AT_RISK_INTEREST_DAMP = 0.08
# HS activity counts, range 3-6, mean 4.15 (specs §4.1)
ACT_COUNT_W = [0.30, 0.35, 0.25, 0.10]


def choices(seq, weights):
    return rng.choices(seq, weights=weights, k=1)[0]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm01(values):
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return [(v - lo) / span for v in values]


def zipf_weights(n, exponent):
    return [1.0 / ((i + 1) ** exponent) for i in range(n)]


# ------------------------------------------------------------ load DR-1
def load_pool():
    with open(cfg.pool_path(), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ DR-4 alumni
def gen_alumni():
    rows = []
    for i in range(cfg.N_ALUM):
        z = rng.gauss(0, 1)
        rating = int(clamp(round(3.05 + 0.78 * z), 1, 5))
        gpa = round(clamp(3.62 + 0.235 * z + rng.gauss(0, 0.07), 2.55, 4.30), 2)
        submitted = rng.random() < 0.62
        sat = int(clamp(round((1288 + 88 * z + rng.gauss(0, 38)) / 10) * 10,
                        1010, 1600)) if submitted else ""
        school = choices(list(SCHOOL_MAJORS), SCHOOL_W)
        entry_major = rng.choice(SCHOOL_MAJORS[school])
        sport = rng.choice(V.SPORTS) if rng.random() < 0.09 else ""
        seasons = rng.randint(1, 4) if sport else 0

        # thriving latent, partly explained by entry profile (DR-4 signal)
        t = ENTRY_OUTCOME_LOAD * z + math.sqrt(
            max(0.0, 1 - ENTRY_OUTCOME_LOAD ** 2)) * rng.gauss(0, 1)
        engage = rng.gauss(0.35 * t, 0.9)

        fy_gpa = round(clamp(3.05 + 0.30 * t + rng.gauss(0, 0.22), 1.4, 4.0), 2)
        final = round(clamp(fy_gpa + 0.16 + 0.10 * t + rng.gauss(0, 0.16),
                            1.5, 4.0), 2)
        delta = round(final - fy_gpa, 2)
        retained = "Y" if rng.random() < clamp(0.91 + 0.05 * t, 0.55, 0.995) else "N"
        grad4 = "Y" if retained == "Y" and rng.random() < clamp(
            0.82 + 0.06 * t, 0.35, 0.98) else "N"
        grad6 = "Y" if grad4 == "Y" or (
            retained == "Y" and rng.random() < 0.55) else "N"
        probation = "Y" if rng.random() < clamp(0.12 - 0.05 * t, 0.01, 0.45) else "N"
        leadership = int(clamp(round(1.1 + 0.9 * engage + rng.gauss(0, 0.7)), 0, 6))
        captain = "Y" if seasons >= 2 and rng.random() < 0.28 else "N"
        thesis = "Y" if rng.random() < clamp(0.18 + 0.12 * t, 0.02, 0.62) else "N"
        latin = "Y" if final >= 3.65 and rng.random() < 0.7 else "N"

        rows.append({
            "alum_id": f"L{2019 + i % 7}-{10000 + i}",
            "cohort_year": 2019 + i % 7,
            # ---- entry snapshot, mirrors DR-1 (DR-4a)
            "entry_academic_rating": rating,
            "entry_hs_gpa_recalc": f"{gpa:.2f}",
            "entry_testing_policy": "Submitted" if submitted else "Test Optional",
            "entry_sat_total": sat,
            "entry_region": choices(REGIONS, REGION_W),
            "entry_hs_type": choices(HS_TYPES, HS_W),
            "entry_first_generation": "Y" if rng.random() < 0.21 else "N",
            "entry_family_income_band": choices(INCOME, INCOME_W),
            "entry_institutional_aid": int(round(
                rng.uniform(0, 62000) / 100) * 100),
            "entry_intended_school": school,
            "entry_intended_major": entry_major,
            "entry_athletic_recruit_sport": sport,
            "entry_demonstrated_interest_rating": int(
                clamp(round(rng.gauss(3.1, 1.1)), 1, 5)),
            # ---- outcome block
            "final_school": school,
            "final_major": entry_major if rng.random() < 0.72
            else rng.choice(SCHOOL_MAJORS[school]),
            "first_year_gpa": f"{fy_gpa:.2f}",
            "final_cum_gpa": f"{final:.2f}",
            "gpa_delta": f"{delta:.2f}",
            "retained_year2": retained,
            "graduated_4yr": grad4,
            "graduated_6yr": grad6,
            "academic_probation_ever": probation,
            "major_changes": 0 if rng.random() < 0.55 else rng.randint(1, 3),
            "leadership_roles_count": leadership,
            "varsity_seasons": seasons,
            "team_captain": captain,
            "honors_thesis": thesis,
            "latin_honors": latin,
            "internships_count": int(clamp(round(rng.gauss(1.4 + 0.4 * t, 0.9)), 0, 4)),
            "study_abroad": "Y" if rng.random() < clamp(0.22 + 0.06 * t, 0.03, 0.55) else "N",
            "postgrad_status_6mo": choices(POSTGRAD, POSTGRAD_W),
            "_t": t,
        })

    # ---- four success components, each normalised to [0,1] (DR-4c)
    attain = norm01([float(r["final_cum_gpa"]) for r in rows])
    growth = norm01([float(r["gpa_delta"]) for r in rows])
    contrib = norm01([r["leadership_roles_count"] + 1.5 * (r["team_captain"] == "Y")
                      + 1.5 * (r["honors_thesis"] == "Y") for r in rows])
    for r, a, g, c in zip(rows, attain, growth, contrib):
        completion = (0.45 * (r["retained_year2"] == "Y")
                      + 0.40 * (r["graduated_4yr"] == "Y")
                      + 0.15 * (r["academic_probation_ever"] == "N"))
        r["success_attainment"] = round(a, 4)
        r["success_growth"] = round(g, 4)
        r["success_completion"] = round(completion, 4)
        r["success_contribution"] = round(c, 4)
        r["_composite"] = (a + g + completion + c) / 4

    ranked = sorted(rows, key=lambda r: -r["_composite"])
    for pos, r in enumerate(ranked):
        r["exemplar_flag"] = "Y" if pos < 200 else "N"
        q = pos / len(ranked)
        r["success_composite_tier"] = (
            "Tier 1 - Thrived" if q < 0.20 else
            "Tier 2 - Strong" if q < 0.45 else
            "Tier 3 - Solid" if q < 0.72 else
            "Tier 4 - Struggled")
    for r in rows:
        r.pop("_t"), r.pop("_composite")
    return rows


# --------------------------------------------------- DR-2 activity edges
def gen_activity_edges(pool, alumni):
    weights = zipf_weights(len(V.HS_ELIGIBLE), V.ZIPF_EXPONENT)
    order = V.HS_ELIGIBLE[:]
    rng.shuffle(order)          # which activities become hubs is seeded, not alphabetical
    act_pool, act_w = order, weights

    edges = []

    def emit(pid, ptype, aid, context):
        edges.append({
            "person_id": pid, "person_type": ptype, "activity_id": aid,
            "context": context,
            "role": choices(list(V.ROLE_DIST), list(V.ROLE_DIST.values())),
            "years_involved": rng.randint(1, 4),
            "recognition_level": choices(list(V.RECOG_DIST),
                                         list(V.RECOG_DIST.values())),
        })

    def pick_set(n, forced):
        chosen = list(forced)
        guard = 0
        while len(chosen) < n and guard < 60:
            guard += 1
            a = rng.choices(act_pool, weights=act_w, k=1)[0]
            if a not in chosen:
                chosen.append(a)
        return chosen

    for r in pool:
        forced = []
        tag = r.get("special_talent_tag", "")
        if tag in TALENT_TO_ACT:
            forced.append(TALENT_TO_ACT[tag])
        n = rng.choices([3, 4, 5, 6], weights=ACT_COUNT_W, k=1)[0]
        for aid in pick_set(n, forced):
            emit(r["applicant_id"], "Applicant", aid, "High School")
        sport = r.get("athletic_recruit_sport", "")
        if sport:
            edges.append({
                "person_id": r["applicant_id"], "person_type": "Applicant",
                "activity_id": V.SPORT_TO_ACT[sport], "context": "High School",
                "role": "Captain" if r.get("coach_support_tier") == "Slot"
                        else "Member",
                "years_involved": rng.randint(2, 4),
                "recognition_level": choices(list(V.RECOG_DIST),
                                             list(V.RECOG_DIST.values())),
            })

    for r in alumni:
        n = rng.choices([3, 4, 5, 6], weights=ACT_COUNT_W, k=1)[0]
        for aid in pick_set(n, []):
            emit(r["alum_id"], "Alum", aid, "High School")
        for aid in pick_set(rng.randint(1, 3), []):
            emit(r["alum_id"], "Alum", aid, "Undergraduate")
        if r["entry_athletic_recruit_sport"]:
            edges.append({
                "person_id": r["alum_id"], "person_type": "Alum",
                "activity_id": V.SPORT_TO_ACT[r["entry_athletic_recruit_sport"]],
                "context": "Undergraduate",
                "role": "Captain" if r["team_captain"] == "Y" else "Member",
                "years_involved": max(1, int(r["varsity_seasons"])),
                "recognition_level": choices(list(V.RECOG_DIST),
                                             list(V.RECOG_DIST.values())),
            })
    return edges


# ------------------------------------------------------ DR-6 club edges
def gen_club_edges(pool, alumni, act_edges):
    by_person: dict[str, set[str]] = {}
    for e in act_edges:
        by_person.setdefault(e["person_id"], set()).add(e["activity_id"])

    edges = []
    base_w = {c: (AT_RISK_INTEREST_DAMP if c in V.AT_RISK_REQUIRED else 1.0)
              for c in V.CLUB_IDS}

    def sample_clubs(done_activities, n):
        if n == 0:
            return []
        w = [base_w[c] * (LINKED_INTEREST_PULL
                          if V.LINKED[c] and V.LINKED[c] in done_activities
                          else 1.0)
             for c in V.CLUB_IDS]
        picked, guard = [], 0
        while len(picked) < n and guard < 80:
            guard += 1
            c = rng.choices(V.CLUB_IDS, weights=w, k=1)[0]
            if c not in picked:
                picked.append(c)
        return picked

    for r in pool:
        n = rng.choices([0, 1, 2, 3, 4], weights=INTEREST_COUNT_W, k=1)[0]
        for cid in sample_clubs(by_person.get(r["applicant_id"], set()), n):
            edges.append({
                "person_id": r["applicant_id"], "person_type": "Applicant",
                "club_id": cid, "relationship": "INTERESTED_IN",
                "interest_signal": choices(list(V.SIGNAL_DIST),
                                           list(V.SIGNAL_DIST.values())),
                "signal_strength": choices(list(V.STRENGTH_DIST),
                                           list(V.STRENGTH_DIST.values())),
                "club_role": "", "years_active": "",
            })

    for r in alumni:
        n = rng.choices([0, 1, 2, 3, 4], weights=MEMBER_COUNT_W, k=1)[0]
        for cid in sample_clubs(by_person.get(r["alum_id"], set()), n):
            edges.append({
                "person_id": r["alum_id"], "person_type": "Alum",
                "club_id": cid, "relationship": "MEMBER_OF",
                "interest_signal": "", "signal_strength": "",
                "club_role": choices(list(V.CLUB_ROLE_DIST),
                                     list(V.CLUB_ROLE_DIST.values())),
                "years_active": rng.randint(1, 4),
            })
    return edges


# ------------------------------------------- DR-6 club dimension rosters
def gen_clubs(pool, club_edges):
    """Roster numbers are set AFTER edges exist, so DR-6c holds by construction."""
    yield_by = {r["applicant_id"]: float(r["predicted_yield_prob"]) for r in pool}
    pipeline: dict[str, float] = {c: 0.0 for c in V.CLUB_IDS}
    heads: dict[str, int] = {c: 0 for c in V.CLUB_IDS}
    for e in club_edges:
        if e["relationship"] == "INTERESTED_IN":
            pipeline[e["club_id"]] += yield_by.get(e["person_id"], 0.0)
            heads[e["club_id"]] += 1

    # SR-8b needs at-risk clubs with a *thin* pipeline, 1 or 2 interested.
    # Zero interested is not a sole pipeline, it is no pipeline, and gives
    # IR-7e nothing to surface.
    sole = sorted((c for c in V.AT_RISK_REQUIRED if 1 <= heads[c] <= 2),
                  key=lambda c: heads[c])

    rows = []
    for cid, name, category, linked, proxy, want_risk in V.CLUBS:
        min_viable = rng.randint(8, 16)
        current = rng.randint(min_viable, min_viable + 12)
        pipe = pipeline[cid]
        if want_risk:
            # force current - graduating + pipeline < min_viable
            graduating = int(current + pipe - min_viable) + rng.randint(2, 5)
            graduating = clamp(graduating, 1, current)
            status = "Probationary" if rng.random() < 0.5 else "Active"
        else:
            graduating = int(current * rng.uniform(0.12, 0.32))
            # keep healthy clubs healthy
            while current - graduating + pipe < min_viable * 1.05:
                current += 2
            status = "Active"
        rows.append({
            "club_id": cid, "club_name": name, "category": category,
            "linked_activity_id": linked or "",
            "current_members": current,
            "graduating_members": graduating,
            "min_viable_members": min_viable,
            "charter_status": status,
            "advisor_backed": "Y" if rng.random() < 0.82 else "N",
            "proxy_sensitive": "Y" if proxy else "N",
        })
    return rows, sole


# ------------------------------------------------------------------ write
def write(path: Path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path.name:28s} {len(rows):6d} rows  {len(fields)} cols")


def main():
    cfg.DATA_DIR.mkdir(exist_ok=True)
    pool = load_pool()
    print(f"loaded DR-1 pool: {len(pool)} applicants")

    write(cfg.DATA_DIR / "activities_dim.csv",
          [{"activity_id": a, "activity_name": n, "category": c}
           for a, n, c in V.ACTIVITIES])

    alumni = gen_alumni()
    act_edges = gen_activity_edges(pool, alumni)
    club_edges = gen_club_edges(pool, alumni, act_edges)
    clubs, sole = gen_clubs(pool, club_edges)

    write(cfg.DATA_DIR / "campus_clubs_dim.csv", clubs)
    write(cfg.DATA_DIR / "alumni_outcomes.csv", alumni)
    write(cfg.DATA_DIR / "activities_edges.csv", act_edges)
    write(cfg.DATA_DIR / "club_edges.csv", club_edges)

    # ---------------------------------------------------------- report
    import statistics as st
    print("\nchecks")
    hs = {}
    ug = {}
    for e in act_edges:
        tgt = hs if e["context"] == "High School" else ug
        tgt[e["person_id"]] = tgt.get(e["person_id"], 0) + 1
    print(f"  HS activity edges per person mean {st.mean(hs.values()):.2f} "
          f"(spec 3-6, mean 4.2)")
    print(f"  undergrad edges per alum mean     {st.mean(ug.values()):.2f}")

    ic = {r["applicant_id"]: 0 for r in pool}
    mc = {r["alum_id"]: 0 for r in alumni}
    for e in club_edges:
        (ic if e["relationship"] == "INTERESTED_IN" else mc)[e["person_id"]] += 1
    print(f"  applicant club interests mean {st.mean(ic.values()):.2f}  "
          f"zero-share {sum(v == 0 for v in ic.values()) / len(ic):.0%}")
    print(f"  alum club memberships mean   {st.mean(mc.values()):.2f}")

    yb = {r["applicant_id"]: float(r["predicted_yield_prob"]) for r in pool}
    pipe = {c["club_id"]: 0.0 for c in clubs}
    heads = {c["club_id"]: 0 for c in clubs}
    for e in club_edges:
        if e["relationship"] == "INTERESTED_IN":
            pipe[e["club_id"]] += yb.get(e["person_id"], 0)
            heads[e["club_id"]] += 1
    at_risk = [c for c in clubs
               if c["current_members"] - c["graduating_members"]
               + pipe[c["club_id"]] < c["min_viable_members"]]
    print(f"  clubs at succession risk     {len(at_risk)} of {len(clubs)} "
          f"({len(at_risk) / len(clubs):.0%})  DR-6c needs >= 9")
    for c in at_risk:
        proj = c["current_members"] - c["graduating_members"] + pipe[c["club_id"]]
        print(f"      {c['club_id']} {c['club_name'][:30]:32s} "
              f"proj {proj:5.1f} < min {c['min_viable_members']:2d}  "
              f"interested {heads[c['club_id']]:3d}")
    print(f"  at-risk w/ sole pipeline     {len(sole)} {sole}  (SR-8b, AC-25)")
    print(f"  proxy_sensitive clubs        "
          f"{sum(c['proxy_sensitive'] == 'Y' for c in clubs)}")
    print(f"  exemplars                    "
          f"{sum(r['exemplar_flag'] == 'Y' for r in alumni)}")

    # ---- DR-4 entry-to-outcome signal (specs §4.3 target Spearman ~0.35)
    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) ** 0.5
               * sum((b - my) ** 2 for b in ry) ** 0.5)
        return num / den if den else 0.0

    sp = spearman([r["entry_academic_rating"] for r in alumni],
                  [float(r["final_cum_gpa"]) for r in alumni])
    print(f"  entry rating vs final GPA    Spearman {sp:+.3f}  (target ~0.35)")

    # ---- DR-6d bridge association, Cramer's V pooled over bridged pairs
    did = {}
    for e in act_edges:
        if e["person_type"] == "Applicant":
            did.setdefault(e["person_id"], set()).add(e["activity_id"])
    want = {}
    for e in club_edges:
        if e["relationship"] == "INTERESTED_IN":
            want.setdefault(e["person_id"], set()).add(e["club_id"])
    n11 = n10 = n01 = n00 = 0
    ids = [r["applicant_id"] for r in pool]
    bridged = [(c, V.LINKED[c]) for c in V.CLUB_IDS if V.LINKED[c]]
    for pid in ids:
        acts, clubs_i = did.get(pid, set()), want.get(pid, set())
        for cid, aid in bridged:
            a, b = aid in acts, cid in clubs_i
            if a and b:
                n11 += 1
            elif a:
                n10 += 1
            elif b:
                n01 += 1
            else:
                n00 += 1
    tot = n11 + n10 + n01 + n00
    chi_den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = ((n11 * n00 - n10 * n01) / chi_den ** 0.5) if chi_den else 0.0
    print(f"  HS activity -> linked club   Cramer's V {abs(phi):.3f}  "
          f"(DR-6d target ~0.45, n={tot})")


if __name__ == "__main__":
    main()
