"""Generated explanations (XR-1 to XR-11).

Every sentence here is derived from the SR-1 decomposition. None is
hand-authored, because a hand-written rationale would stop tracking the model
the first time a weight changed.

Two rules keep the output honest rather than promotional:

  - a `despite` clause names the strongest dimension that did NOT support the
    match, so a rationale cannot read as pure advocacy;
  - dropped dimensions are disclosed, so a silent absence of evidence never
    looks like agreement (SR-7b).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg                                   # noqa: E402
from core.loader import Dataset                        # noqa: E402

DRIVER_SHARE = 0.60        # name contributors until this share is covered
DESPITE_BELOW = 0.35       # only mention a weak dimension if it is this poor
LOW_INCOME = {"Under $30K", "$30K - $59K"}


# ------------------------------------------------------------------ XR-2
def strength_word(pct: float) -> str:
    for floor, word in cfg.STRENGTH_BANDS:
        if pct >= floor:
            return word
    return cfg.STRENGTH_BANDS[-1][1]


def _phrase(name: str, detail: str | None = None,
            form: str = "positive") -> str:
    """`form` selects the positive, weak or absent wording for a dimension.

    A single phrase set cannot serve all three. Reusing the positive form in a
    `despite` clause asserted the opposite of the score.
    """
    i = cfg.DIM_NAMES.index(name)
    if form == "weak":
        return cfg.DIM_PHRASES_WEAK[i]
    if form == "absent":
        return cfg.DIM_PHRASES_ABSENT[i]
    text = cfg.DIM_PHRASES[i]
    if "{detail}" in text:
        if not detail:
            # no specific shared item to name, so fall back to the neutral noun
            # rather than interpolating a placeholder into the sentence
            return cfg.DIM_PHRASES_ABSENT[i] + " in common"
        return text.format(detail=detail)
    return text


def _join(items: list[str]) -> str:
    if not items:
        return "nothing in particular"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def shared_detail(ds: Dataset, applicant: str, alum: str) -> dict[str, str]:
    """Names for the {detail} slots: the rarest thing the pair actually share.

    Naming the specific activity is what makes a rationale specific instead of
    generic, and rarity is the right tiebreak because a shared hub says little.
    """
    out: dict[str, str] = {}
    acts = ds.act_sets.get(applicant, set()) & ds.act_sets.get(alum, set())
    if acts:
        best = max(acts, key=lambda a: ds.act_idf.get(a, 0.0))
        out["Activity portfolio"] = ds.activity_name(best)
    clubs = (ds.club_sets.get(applicant, set()) & ds.club_sets.get(alum, set())
             ) - ds.proxy_clubs
    if clubs:
        best = max(clubs, key=lambda c: ds.club_idf.get(c, 0.0))
        out["Campus club affinity"] = ds.club_name(best)
    return out


def rationale(dec: dict, strength_pct: float, w: np.ndarray,
              details: dict[str, str] | None = None) -> str:
    """specs §8 template: strength, drivers, despite, dropped."""
    details = details or {}
    contribs = dec["contributions"]
    matches = dec["matches"]
    total = sum(contribs.values())

    drivers: list[str] = []
    running = 0.0
    for name, c in sorted(contribs.items(), key=lambda kv: -kv[1]):
        if c <= 0:
            continue
        drivers.append(_phrase(name, details.get(name)))
        running += c
        if total > 0 and running / total >= DRIVER_SHARE:
            break

    # never let the despite clause name something already credited as a driver
    named = set()
    running = 0.0
    for name, c in sorted(contribs.items(), key=lambda kv: -kv[1]):
        if c <= 0:
            continue
        named.add(name)
        running += c
        if total > 0 and running / total >= DRIVER_SHARE:
            break

    despite = ""
    eligible = {n: m for n, m in matches.items()
                if m < DESPITE_BELOW and n not in named
                and w[cfg.DIM_NAMES.index(n)] >= cfg.WEIGHT["Moderate"]}
    if eligible:
        worst = min(eligible, key=eligible.get)
        despite = f", despite {_phrase(worst, form='weak')}"

    dropped = ""
    if dec["dropped"]:
        names = _join([_phrase(n, form="absent") for n in dec["dropped"]])
        dropped = f" (no {names} to compare)"

    return f"{strength_word(strength_pct).capitalize()}, driven mainly by " \
           f"{_join(drivers)}{despite}{dropped}."


def pick_rationale(ds: Dataset, applicant_idx: int, priority_pct: dict[str, float],
                   weights: dict[str, float], clubs_carried: list[str] | None,
                   thin: bool, in_list: bool = True) -> str:
    """Why this applicant is, or is not, on the short list, in priority terms."""
    active = {k: v for k, v in weights.items() if v > 0}
    if not active:
        return "No priority is active, so the short list is arbitrary."

    total = sum(active.values())
    share = {k: (v / total) * priority_pct.get(k, 0.0) for k, v in active.items()}
    ordered = sorted(share.items(), key=lambda kv: -kv[1])
    lead = cfg.PRIORITY_LABELS[ordered[0][0]].lower()

    weak = [cfg.PRIORITY_LABELS[k].lower() for k, v in active.items()
            if priority_pct.get(k, 0.0) < 0.30]
    if in_list:
        parts = [f"Shortlisted mainly on {lead}"]
        if len(ordered) > 1 and ordered[1][1] > 0:
            parts.append(
                f"supported by {cfg.PRIORITY_LABELS[ordered[1][0]].lower()}")
    else:
        # strongest available argument, not a claim that they were selected
        parts = [f"Not shortlisted; strongest case would be {lead}"]
    text = ", ".join(parts)
    if weak:
        text += f", and weak on {_join(weak)}"
    if clubs_carried:
        names = _join([ds.club_name(c) for c in clubs_carried])
        text += (f". Sole realistic pipeline for {names}, which is at "
                 f"succession risk")
    if thin:
        text += f". {cfg.THIN_NEIGHBOURHOOD_NOTE}"
    return text + "."


# ------------------------------------------------------------------ XR-9
def config_sentence(priority_words: dict[str, str], size_word: str,
                    club_word: str, dims_off: list[str],
                    detail: str | None = None,
                    clustered: bool | None = None) -> str:
    """Always on screen. Word-scale labels appear verbatim, so the sentence
    never contains a number."""
    by_level: dict[str, list[str]] = {}
    for key, word in priority_words.items():
        by_level.setdefault(word, []).append(cfg.PRIORITY_LABELS[key].lower())

    ranked = [lvl for lvl in reversed(cfg.SCALE) if lvl in by_level]
    clauses: list[str] = []
    for lvl in ranked:
        names = _join(by_level[lvl])
        if lvl == "Not considered":
            clauses.append(f"ignoring {names}")
        elif lvl == "Defining":
            clauses.append(f"treating {names} as decisive")
        else:
            clauses.append(f"weighting {names} as {lvl.lower()}")

    text = "You are " + "; ".join(clauses)
    text += f". Short list breadth is {size_word.lower()}"
    if club_word != "Not considered":
        text += f", with club coverage {club_word.lower()} inside diversity"
    if dims_off:
        text += f". Similarity ignores {_join([d.lower() for d in dims_off])}"
    # XR-13: an edgeless view must never be ambiguous between "no edges exist"
    # and "edges are not being drawn"
    if detail is not None:
        text += f". The graph is showing {detail.lower()}"
        if clustered is not None:
            text += (", grouped into cohort clusters" if clustered
                     else ", with cohort clustering off")
    return text + "."


# ------------------------------------------------------------------ XR-4
def attribute_change(prev_words: dict[str, str], curr_words: dict[str, str],
                     prev_mask: np.ndarray, curr_mask: np.ndarray,
                     recompute, ds: Dataset) -> dict:
    """Which applicants moved, and which single control caused it.

    Revert one priority at a time; the single revert that flips membership is
    the attributed cause. If no single revert flips it, report a combined
    effect. A graph that silently rearranges teaches nothing.
    """
    entered = np.flatnonzero(curr_mask & ~prev_mask)
    left = np.flatnonzero(prev_mask & ~curr_mask)
    changed = [k for k in curr_words if prev_words.get(k) != curr_words[k]]

    cause: dict[int, str] = {}
    if changed and (entered.size or left.size):
        for key in changed:
            probe = dict(curr_words)
            probe[key] = prev_words.get(key, curr_words[key])
            mask = recompute(probe)
            for i in entered:
                if not mask[i] and i not in cause:
                    cause[int(i)] = cfg.PRIORITY_LABELS.get(key, key)
            for i in left:
                if mask[i] and i not in cause:
                    cause[int(i)] = cfg.PRIORITY_LABELS.get(key, key)

    def rows(idx):
        return [{"applicant_id": ds.app_ids[int(i)],
                 "cause": cause.get(int(i), "Combined effect")} for i in idx]

    return {"entered": rows(entered), "left": rows(left),
            "changed_controls": [cfg.PRIORITY_LABELS.get(k, k) for k in changed],
            "n_entered": int(entered.size), "n_left": int(left.size)}


# ------------------------------------------------------------------ XR-5
def cost_of_choice(ds: Dataset, mask: np.ndarray) -> dict:
    """The realised profile of a short list against the whole-pool baseline.

    This is the requirement that keeps the tool advisory rather than persuasive.
    Maxing net revenue must visibly report what it cost.
    """
    app = ds.applicants

    def profile(sel) -> dict:
        sub = app[sel] if sel is not None else app
        n = max(1, len(sub))
        return {
            "n": int(len(sub)),
            "academic": float(np.asarray(sub.academic_rating, dtype=float).mean()),
            "yield": float(np.asarray(sub.predicted_yield_prob, dtype=float).mean()),
            "net_price": float(np.asarray(sub.net_price, dtype=float).mean()),
            "first_gen": float((np.asarray(sub.first_generation) == "Y").sum() / n),
            "low_income": float(sum(
                1 for b in np.asarray(sub.family_income_band)
                if b in LOW_INCOME) / n),
            "regions": int(len(set(np.asarray(sub.region).tolist()))),
        }

    sel = profile(mask)
    base = profile(None)
    deltas = {k: (sel[k] - base[k]) for k in sel if k != "n"}
    return {"shortlist": sel, "baseline": base, "delta": deltas}


# ------------------------------------------------------------------ XR-8
def counterfactual(applicant_idx: int, curr_words: dict[str, str],
                   recompute, in_list: bool) -> str | None:
    """Smallest single word-scale step on one priority that flips membership.

    Computed on demand for the selected applicant only: twenty evaluations at
    most, rather than for the whole pool.
    """
    for key in cfg.PRIORITY_KEYS:
        current = curr_words.get(key, "Moderate")
        here = cfg.SCALE.index(current)
        # try nearest steps first, so the reported change is the smallest one
        for step in sorted(range(len(cfg.SCALE)), key=lambda s: abs(s - here)):
            if step == here:
                continue
            probe = dict(curr_words)
            probe[key] = cfg.SCALE[step]
            mask = recompute(probe)
            if bool(mask[applicant_idx]) != in_list:
                label = cfg.PRIORITY_LABELS[key]
                verb = "leaves" if in_list else "enters"
                return (f"{verb.capitalize()} the short list if {label.lower()} "
                        f"moves from {current.lower()} to "
                        f"{cfg.SCALE[step].lower()}.")
    return None


# ----------------------------------------------------------------- XR-10
def export_payload(ds: Dataset, applicant_idx: int, priority_words: dict[str, str],
                   dim_words: dict[str, str], neighbours: list[tuple[str, float, str]],
                   in_list: bool, pick_note: str, cost: dict) -> dict:
    """Every export carries the configuration that produced it, so no number
    travels without its reasoning (FR-4)."""
    return {
        "applicant_id": ds.app_ids[applicant_idx],
        "on_short_list": bool(in_list),
        "rationale": pick_note,
        "priority_configuration": dict(priority_words),
        "similarity_configuration": dict(dim_words),
        "comparable_alumni": [
            {"alum_id": a, "similarity": round(float(s), 4), "why": why}
            for a, s, why in neighbours],
        "short_list_profile": cost["shortlist"],
        "pool_baseline": cost["baseline"],
        "advisory_notice": (
            "Advisory only. This is a discussion set, not an admission "
            "decision, and it carries no recommendation to admit or deny."),
        "excluded_from_scoring": cfg.PROTECTED_COLUMNS,
    }


if __name__ == "__main__":
    from core.clubs import coverage_matrix, project_rosters, sole_pipeline_applicants
    from core.features import build_tensors, flatten
    from core.loader import load
    from core.priorities import (blend_individual, build_shortlist,
                                 composition_matrix, p1_yield, p2_expected_ntr,
                                 p4_academic, percentile)
    from core.similarity import contract, decompose, topk, weights_from_words

    ds = load()
    M, A = build_tensors(ds)
    M2, A2 = flatten(M, A)
    w = weights_from_words(cfg.DIM_DEFAULTS)
    sim = contract(M2, A2, w, (ds.n_app, ds.n_alum))
    idx, val = topk(sim, cfg.TOPK_EDGES, cfg.SIM_FLOOR)

    p1, p2 = percentile(p1_yield(ds)), percentile(p2_expected_ntr(ds))
    p4, thin, nbrs, parts = p4_academic(sim, ds)
    C, target, labels = composition_matrix(ds)
    ros = project_rosters(ds)
    B, club_ids = coverage_matrix(ds, ros)
    id_rank = np.argsort(np.argsort(np.asarray(ds.app_ids)))

    def recompute(words: dict[str, str]) -> np.ndarray:
        indiv = blend_individual(p1, p2, p4, cfg.WEIGHT[words["w_yield"]],
                                 cfg.WEIGHT[words["w_ntr"]],
                                 cfg.WEIGHT[words["w_acad"]])
        return build_shortlist(indiv, C, target, B, None,
                               cfg.WEIGHT[words["w_div"]], 0.25,
                               cfg.CLASS_TARGET, id_rank)

    print("=" * 70)
    print("XR-2  edge rationales")
    print("=" * 70)
    for a_i in (0, 5, 11):
        e_i = int(idx[a_i, 0])
        dec = decompose(M, A, w, a_i, e_i)
        pct = float((sim[a_i] <= sim[a_i, e_i]).mean())
        det = shared_detail(ds, ds.app_ids[a_i], ds.alum_ids[e_i])
        print(f"\n{ds.app_ids[a_i]} -> {ds.alum_ids[e_i]} (score {dec['score']:.3f})")
        print(f"  {rationale(dec, pct, w, det)}")

    print()
    print("=" * 70)
    print("XR-9  configuration sentence")
    print("=" * 70)
    for name, preset in list(cfg.PRESETS.items())[:3]:
        print(f"\n{name}:")
        print("  " + config_sentence(preset, "Selective", "Minor",
                                     ["Socioeconomic context"]))

    print()
    print("=" * 70)
    print("XR-5  cost of the choice")
    print("=" * 70)
    for name in ("Balanced", "Revenue-first", "Access and mission"):
        mask = recompute(cfg.PRESETS[name])
        c = cost_of_choice(ds, mask)
        s, d = c["shortlist"], c["delta"]
        print(f"{name:20s} acad {s['academic']:.2f} ({d['academic']:+.2f})  "
              f"yield {s['yield']:.3f} ({d['yield']:+.3f})  "
              f"net ${s['net_price']:>6,.0f} ({d['net_price']:+,.0f})  "
              f"lowinc {s['low_income']:.0%} ({d['low_income']:+.0%})")

    print()
    print("=" * 70)
    print("XR-4  what changed")
    print("=" * 70)
    for label, prev, curr in [
        ("single control: academic strength Moderate -> Defining",
         cfg.PRESETS["Balanced"],
         {**cfg.PRESETS["Balanced"], "w_acad": "Defining"}),
        ("preset jump: Balanced -> Revenue-first",
         cfg.PRESETS["Balanced"], cfg.PRESETS["Revenue-first"]),
    ]:
        ch = attribute_change(prev, curr, recompute(prev), recompute(curr),
                              recompute, ds)
        print(f"\n{label}")
        print(f"  controls changed: {ch['changed_controls']}")
        print(f"  entered {ch['n_entered']}, left {ch['n_left']}")
        for r in ch["entered"][:3]:
            print(f"    + {r['applicant_id']}  cause: {r['cause']}")
        for r in ch["left"][:2]:
            print(f"    - {r['applicant_id']}  cause: {r['cause']}")

    print()
    print("=" * 70)
    print("XR-8  counterfactual, and XR-2 pick rationale")
    print("=" * 70)
    words = cfg.PRESETS["Balanced"]
    mask = recompute(words)
    carried = sole_pipeline_applicants(ds, ros)
    out = 0
    for a_i in range(ds.n_app):
        if mask[a_i]:
            continue
        cf = counterfactual(a_i, words, recompute, False)
        if cf:
            pid = ds.app_ids[a_i]
            pp = {"w_yield": float(p1[a_i]), "w_ntr": float(p2[a_i]),
                  "w_acad": float(p4[a_i]), "w_div": 0.5}
            note = pick_rationale(ds, a_i, pp,
                                  {k: cfg.WEIGHT[v] for k, v in words.items()},
                                  carried.get(pid), bool(thin[a_i]),
                                  in_list=False)
            print(f"\n{pid} (not shortlisted)")
            print(f"  {cf}")
            print(f"  {note}")
            out += 1
            if out == 3:
                break
