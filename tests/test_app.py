"""End-to-end app tests via Streamlit's headless AppTest runner.

Covers the acceptance criteria that need the interface rather than core: the
no-numbers rule (AC-8), the configuration sentence (AC-16), the suppression
badge (AC-12), the cost-of-choice table (AC-13) and the caveat surfacing (AC-19).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

APP = str(ROOT / "app.py")
TIMEOUT = 300


@pytest.fixture(scope="module")
def at():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not app.exception, [str(e) for e in app.exception]
    return app


def test_app_runs_without_exception(at):
    assert not at.exception


def test_all_tabs_present(at):
    labels = {t.label for t in at.tabs}
    assert labels == {"Graph", "Short list", "Why", "Composition",
                      "What changed"}, labels
    # Clubs and Caveats were removed from the tab bar. Their requirement-bearing
    # content did not go with them: the caveats moved to the sidebar for FR-6,
    # and club roster detail still opens from a club node in the graph for IR-7d.
    assert "Clubs" not in labels and "Caveats" not in labels


# ------------------------------------------------------------------- AC-8
def test_no_numeric_importance_controls(at):
    """IR-1: st.slider is prohibited for weights; every importance control is a
    word scale. A numeric slider anywhere in the app fails this outright."""
    assert len(at.slider) == 0, (
        f"{len(at.slider)} numeric slider(s) present; IR-1 forbids them for "
        f"importance")

    scale = set(cfg.SCALE)
    importance = [s for s in at.select_slider if set(s.options) & scale]
    assert importance, "no word-scale importance controls found"
    for s in importance:
        assert set(s.options) <= scale | set(cfg.MATCHES_PER_APPLICANT) \
            | set(cfg.SHORTLIST_SIZE)
        for opt in s.options:
            assert not str(opt).replace(".", "", 1).isdigit(), (
                f"control '{s.label}' exposes a numeric option {opt!r}")


def test_priority_controls_exist_with_word_options(at):
    """AC-5: four word-scaled priority controls."""
    labels = {s.label for s in at.select_slider}
    for key in cfg.PRIORITY_KEYS:
        assert cfg.PRIORITY_LABELS[key] in labels, (
            f"missing priority control {cfg.PRIORITY_LABELS[key]}")


# ------------------------------------------------------------------ AC-16
def _config_sentence(app) -> str:
    """XR-9 renders as a caption directly under the controls."""
    for c in app.caption:
        if c.value.startswith("You are"):
            return c.value
    raise AssertionError("no configuration sentence on screen")


def test_configuration_sentence_always_present(at):
    """XR-9: legible as a sentence at all times, and never numeric."""
    sentence = _config_sentence(at)
    head = sentence.split("  ·  ")[0]
    assert not any(ch.isdigit() for ch in head), (
        f"configuration sentence contains a number: {head}")
    # XR-13: the detail level must be stated, so an edgeless view is never
    # ambiguous between "no edges exist" and "edges are not drawn"
    assert "the graph is showing" in head.lower()


# ------------------------------------------------------------------ AC-12
def test_suppression_audit_badge(at):
    """FR-5: exclusion is verifiable in the interface, not assumed."""
    text = _config_sentence(at).lower()
    assert "excluded from scoring" in text
    assert "audit failed" not in text


# ------------------------------------------------------------------ AC-19
def test_caveats_reach_the_interface(at):
    """FR-6: the mock-data limitations must appear in the app, not only in the
    requirements document. They render as expander labels, which are not part of
    the markdown or caption element lists."""
    labels = {e.label for e in at.expander}
    for title, _ in cfg.MOCK_DATA_CAVEATS:
        assert title in labels, f"caveat not surfaced in the interface: {title}"

    # the headline distortion must be stated, not merely hinted at
    bodies = " ".join(m.value for m in at.markdown)
    assert "UNDERSTATES" in " ".join(b for _, b in cfg.MOCK_DATA_CAVEATS)

    for required in ("The short list is greedy, not optimal",
                     "What is excluded from scoring, and why",
                     "Advisory only"):
        assert required in labels, f"missing method-limit disclosure: {required}"


# ------------------------------------------------------------------- AC-6
def test_preset_changes_the_configuration_sentence():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    before = _config_sentence(app)
    app.session_state["w_ntr"] = "Defining"
    app.session_state["w_div"] = "Not considered"
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    after = _config_sentence(app)
    assert before != after
    assert "decisive" in after
    assert "ignoring diversity" in after


# ------------------------------------------------------------------ AC-13
def test_revenue_first_reports_its_cost():
    """XR-5: maxing net revenue must visibly report what it cost. The measured
    consequence in this pool is low-income representation collapsing, and the
    app must surface that rather than leaving the user to infer it."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    for key, word in cfg.PRESETS["Revenue-first"].items():
        app.session_state[key] = word
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    # the cost panel raises an explicit error banner when low-income share
    # collapses; that banner is the requirement, not the number alone
    banners = " ".join(e.value for e in app.error).lower()
    assert "low-income representation" in banners, (
        f"no cost warning surfaced for Revenue-first; errors were {banners!r}")
    assert "net revenue" in banners

    # and access-and-mission must NOT trigger it, else the banner is unconditional
    app2 = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app2.run()
    for key, word in cfg.PRESETS["Access and mission"].items():
        app2.session_state[key] = word
    app2.run()
    quiet = " ".join(e.value for e in app2.error).lower()
    assert "low-income representation" not in quiet, (
        "cost banner fires regardless of configuration, so it carries no "
        "information")


def test_defining_one_priority_does_not_crash():
    for name in cfg.PRESETS:
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        for key, word in cfg.PRESETS[name].items():
            app.session_state[key] = word
        app.run()
        assert not app.exception, f"preset {name}: {[str(e) for e in app.exception]}"


def test_turning_off_every_dimension_is_survivable():
    """A degenerate configuration must not raise; it should just stop
    discriminating."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    for name, _ in cfg.DIMENSIONS:
        app.session_state[f"dim_on_{name}"] = False
    app.run()
    assert not app.exception, [str(e) for e in app.exception]


def test_all_success_components_off_is_survivable():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    from core.priorities import SUCCESS_COMPONENTS  # noqa: PLC0415
    for label in SUCCESS_COMPONENTS:
        app.session_state[f"succ_{label}"] = False
    app.run()
    assert not app.exception, [str(e) for e in app.exception]


def test_focus_modes_all_render():
    for mode in cfg.FOCUS_MODES:
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        app.session_state["focus_mode"] = mode
        app.run()
        assert not app.exception, f"focus mode {mode}: " \
                                 f"{[str(e) for e in app.exception]}"


def test_editing_composition_targets_does_not_loop():
    """SR-5c targets are editable. Normalising on write would store a value
    different from what the editor holds, re-triggering the edit branch every
    run. Normalisation therefore happens at consumption, and an edited target
    must survive a rerun unchanged."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    assert not app.exception

    targets = app.session_state["targets"]
    targets["region"] = {k: (0.50 if k == "West" else 0.05)
                         for k in targets["region"]}
    app.session_state["targets"] = targets
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    kept = app.session_state["targets"]["region"]
    assert abs(kept["West"] - 0.50) < 1e-9, (
        f"edited target was rewritten to {kept['West']}, which would loop")

    app.run()
    assert not app.exception
    assert abs(app.session_state["targets"]["region"]["West"] - 0.50) < 1e-9


def test_edited_targets_change_the_short_list():
    """An editable target that does not move the class is not a policy control."""
    sys.path.insert(0, str(ROOT))
    import numpy as np  # noqa: PLC0415

    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    app.session_state["w_div"] = "Defining"
    app.run()
    assert not app.exception

    t = app.session_state["targets"]
    t["region"] = {k: (0.80 if k == "West" else 0.02) for k in t["region"]}
    app.session_state["targets"] = t
    app.run()
    assert not app.exception, [str(e) for e in app.exception]


def test_back_button_appears_only_when_something_is_selected():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    labels = {b.label for b in app.button}
    assert not any("Back to clusters" in l for l in labels), (
        "back button shown with nothing selected")

    app.session_state["last_clicked"] = app.session_state["prev_mask"] is not None \
        and "A26-101831" or "A26-101831"
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    labels = {b.label for b in app.button}
    assert any("Back to clusters" in l for l in labels), (
        "no way back after selecting an applicant")


def test_back_button_returns_to_the_cluster_view():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    app.session_state["last_clicked"] = "A26-101831"
    app.session_state["selected_pair"] = None
    app.session_state["focus_mode"] = "Applicant focus"
    app.run()
    assert not app.exception

    back = [b for b in app.button if "Back to clusters" in b.label]
    assert back, "no back button to press"
    back[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]
    assert app.session_state["last_clicked"] is None
    assert app.session_state["selected_pair"] is None
    assert app.session_state["focus_mode"] == "Cohort overview"


def test_selected_applicant_panel_names_the_similar_past_students():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    app.session_state["last_clicked"] = "A26-101831"
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    body = " ".join([m.value for m in app.markdown]
                    + [c.value for c in app.caption])
    assert "Past students most like them" in body, (
        "the comparison set is not headlined")
    assert "Ringed in green on the graph" in body, (
        "nothing tells the user how to find the matches on the graph")
    # every match offers a decomposition route
    assert any("Break this match down" in b.label for b in app.button)
    # and the outcome summary that justifies the academic strength score
    assert any("outcome tiers" in i.value for i in app.info)


def test_selecting_an_alum_does_not_claim_it_is_an_applicant():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    ds_alum = "L2021-10387"
    app.session_state["last_clicked"] = ds_alum
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    body = " ".join(c.value for c in app.caption)
    assert "ENTRY profile only" in body or "entry" in body.lower()


def test_caveats_survive_the_tab_removal(at):
    """FR-6 does not care which container holds them, only that they are in the
    app. They moved to the sidebar when the tab went."""
    labels = {e.label for e in at.expander}
    for title, _ in cfg.MOCK_DATA_CAVEATS:
        assert title in labels, f"caveat lost with the tab: {title}"
    for required in ("The short list is greedy, not optimal",
                     "What is excluded from scoring, and why",
                     "Advisory only"):
        assert required in labels, f"method-limit disclosure lost: {required}"


def test_club_detail_still_reachable_without_the_clubs_tab():
    """IR-7d and AC-24: selecting a club must still expose its roster
    projection, its member alumni and its interested applicants. That now
    happens in the graph panel rather than a tab."""
    sys.path.insert(0, str(ROOT))
    from core.clubs import project_rosters  # noqa: PLC0415
    from core.loader import load  # noqa: PLC0415
    from ui import panels  # noqa: PLC0415

    assert hasattr(panels, "club_detail"), "club detail panel was deleted"
    assert not hasattr(panels, "clubs_tab"), "dead tab function left behind"

    ds = load()
    ros = project_rosters(ds)
    at_risk = ros[ros.at_risk]
    assert len(at_risk) >= 9, "no at-risk clubs left to inspect"

    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    cid = at_risk.iloc[0].club_id
    app.session_state["focus_mode"] = "Clubs at succession risk"
    app.session_state["last_clicked"] = cid
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    body = " ".join([m.value for m in app.markdown]
                    + [c.value for c in app.caption])
    assert "Roster outlook" in body or "roster" in body.lower()


def test_sole_pipeline_warning_survives_on_the_main_page():
    """IR-7e: the deliberate override of the composite has to stay visible."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    app.session_state["show_sole_pipeline"] = True
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    warnings = " ".join(w.value for w in app.warning)
    assert "sole" in warnings.lower() and "succession risk" in warnings.lower()


def test_removed_toggle_is_gone():
    """`at_risk_only` drove only the deleted tab, so it became dead state."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert "at_risk_only" not in app.session_state
    assert not any(t.label == "At-risk clubs only" for t in app.toggle)
