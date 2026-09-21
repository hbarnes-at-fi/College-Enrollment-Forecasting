"""Constants for the admissions similarity graph.

Contract: specs.md section 1. No Streamlit imports; this module is shared by
`core/` (headless, tested) and `ui/` (Streamlit).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
POOL_NAME = "mock_admit_pool_fall2026.csv"

# The current-cycle pool (DR-1) is reused unchanged (A1), but it must live INSIDE
# the app directory. Resolving it to ROOT.parent worked locally, where the app is
# a subfolder of the project, and broke the moment the app folder was deployed as
# a repository root: ROOT.parent was then outside the checkout entirely.
# In-app copy first, original sibling layout second, so both work.
POOL_CANDIDATES = [DATA_DIR / POOL_NAME, ROOT.parent / POOL_NAME]


def pool_path() -> Path:
    for candidate in POOL_CANDIDATES:
        if candidate.exists():
            return candidate
    looked = "\n  ".join(str(c) for c in POOL_CANDIDATES)
    raise FileNotFoundError(
        f"Cannot find the admit pool '{POOL_NAME}'. Looked in:\n  {looked}\n"
        f"The app must be self-contained to deploy: keep a copy at "
        f"data/{POOL_NAME} and commit it.")


# Resolved lazily via pool_path(); kept as a module attribute for callers that
# only need the location when it is known to exist.
POOL_PATH = next((c for c in POOL_CANDIDATES if c.exists()), POOL_CANDIDATES[0])

# ---------------------------------------------------------------- scale
SEED = 20260920
N_ALUM = 600
N_DIM = 9
CLASS_TARGET = 285          # A6, matches the aid model
COA = 84_600                # cost of attendance, matches the aid model
EPS = 1e-6

# ------------------------------------------------- word scale -> weight
# IR-1a: monotone increasing, documented here, NEVER rendered in the interface.
# The jump to 2.50 makes `Defining` actually dominate: against three priorities
# at `Moderate` it takes 62% of total weight, against three at `Not considered`
# it takes 100%, which is what AC-6 requires.
WEIGHT = {
    "Not considered": 0.00,
    "Minor": 0.25,
    "Moderate": 0.50,
    "Major": 1.00,
    "Defining": 2.50,
}
SCALE = list(WEIGHT)        # widget options, order preserved

# ------------------------------------------------- DR-2 edge weighting
ROLE_MULT = {
    "Member": 1.0, "Officer": 1.3, "Captain": 1.4,
    "President": 1.5, "Founder": 1.6,
}
RECOG_MULT = {
    "None": 1.0, "School": 1.1, "Regional": 1.25,
    "State": 1.4, "National": 1.6,
}
CLUB_ROLE_MULT = {"Member": 1.0, "Officer": 1.3, "President": 1.5, "Founder": 1.6}
SIGNAL_MULT = {"Passive": 0.6, "Explicit": 1.0, "Sustained": 1.4}

# ---------------------------------------------------------------- knobs
TOPK_EDGES = 3              # NR-3a default
TOPK_P4 = 10                # P4b neighbour count
SIM_FLOOR = 0.25            # NR-3b, and the P4b thin-neighbourhood guard
BRIDGE_BOOST = 1.4          # SR-7a, result capped at 1.0

# ---------------------------------------------------------------- bands
# XR-3: words only, never a number in the interface.
STRENGTH_BANDS = [
    (0.90, "very strong match"),
    (0.75, "strong match"),
    (0.50, "moderate match"),
    (0.25, "limited match"),
    (0.00, "little in common"),
]
# SR-8d: roster projection is an expectation, shown as a band, never a decimal.
# Thresholds are multiples of min_viable_members.
ROSTER_BANDS = [
    ("Secure", 1.25),
    ("Thin", 1.00),
    ("At risk", 0.60),
    ("Critical", 0.00),
]
AT_RISK_BANDS = {"At risk", "Critical"}

SHORTLIST_SIZE = {"Very selective": 0.60, "Selective": 1.00, "Broad": 1.60}
MATCHES_PER_APPLICANT = {"Closest only": 1, "A few": 3, "Several": 6, "Many": 12}
STRICTNESS = {
    "Not considered": 0.10, "Minor": 0.18, "Moderate": 0.25,
    "Major": 0.35, "Defining": 0.50,
}

# ------------------------------------------------- similarity dimensions
# specs.md section 6. Index-aligned with M[:,:,d] and A[:,:,d].
DIMENSIONS = [
    ("Academic preparation", "Major"),
    ("Field of study", "Moderate"),
    ("Activity portfolio", "Major"),
    ("Athletics", "Minor"),
    ("Leadership depth", "Moderate"),
    ("School and geography", "Minor"),
    ("Socioeconomic context", "Not considered"),   # FR-2: off by default
    ("Engagement", "Minor"),
    ("Campus club affinity", "Moderate"),
]
DIM_NAMES = [d[0] for d in DIMENSIONS]
DIM_DEFAULTS = {d[0]: d[1] for d in DIMENSIONS}

# XR-2 phrase fragments, index-aligned with DIMENSIONS.
DIM_PHRASES = [
    "a similar academic profile",
    "the same intended field",
    "a shared {detail} background",
    "shared varsity athletics",
    "comparable leadership experience",
    "a comparable school and region",
    "a similar socioeconomic context",
    "comparable demonstrated interest",
    "shared interest in {detail}",
]

# The `despite` clause needs negative forms. Reusing the positive phrases above
# produced sentences that asserted the opposite of the score, e.g. "despite the
# same intended field" for a pair whose field match was 0.0.
DIM_PHRASES_WEAK = [
    "a different academic profile",
    "a different intended field",
    "little shared activity background",
    "no varsity athletics in common",
    "different leadership experience",
    "a different school and region",
    "a different socioeconomic context",
    "different demonstrated interest",
    "no shared campus club interest",
]

# Used when a dimension is dropped entirely (SR-7b), where there is nothing to
# name because nothing was compared.
DIM_PHRASES_ABSENT = [
    "academic profile",
    "intended field",
    "activity background",
    "varsity athletics",
    "leadership experience",
    "school and region",
    "socioeconomic context",
    "demonstrated interest",
    "campus club interest",
]

THIN_NEIGHBOURHOOD_NOTE = (
    "Academic strength from credentials only; too few comparable alumni "
    "to validate."
)

# ------------------------------------------------- priorities and presets
PRIORITY_KEYS = ["w_yield", "w_ntr", "w_div", "w_acad"]
PRIORITY_LABELS = {
    "w_yield": "Yield",
    "w_ntr": "Net tuition revenue",
    "w_div": "Diversity",
    "w_acad": "Academic strength",
}

PRESETS = {
    "Balanced": dict(w_yield="Moderate", w_ntr="Moderate",
                     w_div="Moderate", w_acad="Moderate"),
    "Yield-first": dict(w_yield="Defining", w_ntr="Minor",
                        w_div="Minor", w_acad="Minor"),
    "Revenue-first": dict(w_yield="Minor", w_ntr="Defining",
                          w_div="Not considered", w_acad="Minor"),
    "Access and mission": dict(w_yield="Minor", w_ntr="Not considered",
                               w_div="Defining", w_acad="Minor"),
    "Academic-first": dict(w_yield="Minor", w_ntr="Minor",
                           w_div="Minor", w_acad="Defining"),
}
PRESET_DESCRIPTIONS = {
    "Balanced": "No priority leads. The blend most institutions claim to run.",
    "Yield-first": "Fill the class with certainty. Trades away academic profile "
                   "and net revenue.",
    "Revenue-first": "Protect net tuition revenue. Historically drives "
                     "low-income representation to zero.",
    "Access and mission": "Maximise composition goals. Costs the most revenue "
                          "of any single-priority setting.",
    "Academic-first": "Strongest measured academic profile. Yields lower and "
                      "discounts more.",
}

# ------------------------------------------------- P3 composition targets
# SR-5c: targets are user-editable and must be visible in the interface.
COMPOSITION_TARGETS = {
    "region": {
        "Mid-Atlantic": 0.34, "New England": 0.11, "Midwest": 0.12,
        "South": 0.15, "Southwest": 0.05, "West": 0.10, "International": 0.13,
    },
    "family_income_band": {
        "Under $30K": 0.12, "$30K - $59K": 0.15, "$60K - $99K": 0.18,
        "$100K - $149K": 0.22, "$150K - $249K": 0.20, "$250K and above": 0.08,
        "Not Reported": 0.05,
    },
    "first_generation": {"Y": 0.24, "N": 0.76},
    "hs_type": {
        "Public": 0.60, "Private - Independent": 0.16, "Parochial": 0.11,
        "Charter": 0.06, "Home School": 0.02, "International School": 0.05,
    },
    "intended_school": {
        "College of Arts & Sciences": 0.42, "School of Engineering": 0.19,
        "School of Business": 0.20, "School of Nursing & Health": 0.13,
        "College of Fine Arts": 0.06,
    },
}

# ------------------------------------------------- protected attributes
# SR-3 / FR-5: dropped at load, never present in `Dataset`.
PROTECTED_COLUMNS = ["ethnicity_ipeds", "gender"]

# ------------------------------------------------- rendering, specs §9
# Symbols carry node type independently of colour (NFR-3). Only colours change
# between themes; shapes never do.
NODE_SYMBOLS = {
    "Applicant": "circle", "Alum": "square", "Activity": "diamond",
    "Athletic team": "triangle-up", "Major": "hexagon", "Campus club": "star",
}
# Shape alone was not carrying the population distinction at these densities:
# a 8px circle and a 9px square are indistinguishable among 1,800 nodes. Size
# now does most of the work, with hue as the backup and shape retained because
# NFR-3 forbids relying on colour alone.
NODE_SIZES = {
    "Applicant": 7, "Alum": 13, "Activity": 6,
    "Athletic team": 10, "Major": 11, "Campus club": 10,
}

DARK = dict(
    paper="#12161C", plot="#12161C",
    # blue vs violet: different hue families, and neither collides with the
    # yellow short-list ring, the green match ring or the red club-risk ring
    node={"Applicant": "#5B8FF9", "Alum": "#C07BF0", "Activity": "#7C8797",
          "Athletic team": "#7C8797", "Major": "#94A0B0",
          "Campus club": "#D8A43C"},
    node_dim="#2E3945",
    edge={"observed_past": "#3A4552", "observed_intent": "#6E5A33",
          "inferred": "#A8496B"},
    cluster_fill="rgba(140,165,195,0.055)",
    cluster_line="rgba(140,165,195,0.22)",
    label="#9FB0C4",
    highlight="#F2F4F7",
    text="#D7DEE6",
)

LIGHT = dict(
    paper="#FCFCFD", plot="#FCFCFD",
    node={"Applicant": "#2C5FB8", "Alum": "#8B3FBF", "Activity": "#8A8A8A",
          "Athletic team": "#8A8A8A", "Major": "#6B6B6B",
          "Campus club": "#B5892B"},
    node_dim="#AEC6D8",
    edge={"observed_past": "#BFBFBF", "observed_intent": "#C7A76B",
          "inferred": "#C2506B"},
    cluster_fill="rgba(120,140,165,0.085)",
    cluster_line="rgba(120,140,165,0.30)",
    label="#54657A",
    highlight="#1A1A1A",
    text="#20262E",
)
THEMES = {"dark": DARK, "light": LIGHT}
DEFAULT_THEME = "dark"

# The short list is the thing a slider moves, so it gets the loudest available
# signal: a yellow ring. Everything not on the list is pushed back hard, because
# a subtle fill change across 755 nodes is invisible.
SHORTLIST_OUTLINE = "#FFD23F"
SHORTLIST_OUTLINE_WIDTH = 2.2
UNPICKED_OPACITY = 0.30

# When one applicant is selected, everything else gets out of the way. The
# layout is frozen (NFR-2) so matched alumni cannot be pulled closer; clarity
# has to come from emphasis and from suppressing everything that is not the
# answer.
FOCUS_APPLICANT_RING = "#7FD8FF"
FOCUS_ALUM_RING = "#5BE59A"
FOCUS_EDGE = "#7FD8FF"
FOCUS_BACKDROP_OPACITY = 0.07     # everything that is not the match
FOCUS_NODE_SIZE = 19              # the selected applicant
FOCUS_ALUM_SIZE = 15              # its matched alumni
FOCUS_RANK_LABELS = ["closest", "2nd closest", "3rd closest",
                     "4th closest", "5th closest"]

# ------------------------------------------------------ grouping and edges
# What the cluster discs represent. Position can encode only one grouping, so
# this is a choice rather than a combination.
GROUPINGS = {
    "Cohort": "Named rule-based cohorts, rarest predicate first (NR-4).",
    "Major": "One disc per intended major, or final major for alumni.",
    "School": "One disc per school, the coarsest grouping.",
}
GROUP_BY_DEFAULT = "Cohort"

# Each person contributes at most this many edges, chosen by importance across
# every edge type rather than a fixed quota per type. Drawing every edge gave
# some nodes a visual degree of eight and others one, which read as noise; a
# uniform degree is what makes the graph look homogeneous.
TOP_EDGES_PER_PERSON = 3

# Major, campus club and athletic status are properties of a person, not things
# in the graph. They were four node types competing for the middle of the canvas
# and contributing edges that said little; as characteristics they read better on
# selection. Activities stay, because a shared activity is what actually links
# two people to each other.
PERSON_KINDS = {"Applicant", "Alum"}
# Nothing sits in the middle any more. With activities gone too, the graph is
# people and the links between them, so connections run person to person rather
# than through a shared hub node. That is what makes the view homogeneous: no
# node has a degree of 170 because everyone attached to it.
CONNECTIVE_KINDS: set[str] = set()
CHARACTERISTIC_KINDS = {"Major", "Athletic team", "Campus club", "Activity"}

# Badge colours for the characteristic keywords shown on selection.
BADGE = {
    "major": ("#1E3A5F", "#8FC4F5"),
    "athlete": ("#5A3A14", "#F5C168"),
    "student": ("#2A3038", "#A8B3C0"),
    "club": ("#3D3212", "#E0C46A"),
    "cohort": ("#243A2E", "#8FE0B0"),
    "activity": ("#2B2440", "#B9A7E8"),
}
ATHLETE_LABEL = "Student-Athlete"
NON_ATHLETE_LABEL = "Student"
# Structural weight by edge kind, multiplied by the rarity of the target. A
# shared rare activity says more than a shared major that half the cluster holds.
EDGE_KIND_WEIGHT = {
    "PLAYS_FOR": 1.35,        # rarest attachment in the data
    "PARTICIPATES_IN": 1.00,
    "MEMBER_OF": 1.05,
    "INTERESTED_IN": 0.80,    # intent, not history
    "STUDIES": 0.70,          # near-universal, so rarely the most informative
}

# Weight on a shared characteristic when projecting person-to-attribute links
# into person-to-person ones. Multiplied by the rarity of the thing shared, so
# two people who both did Quiz Bowl outrank two who both did Volunteering.
SHARED_KIND_WEIGHT = {
    "activity": 1.00,
    "sport": 1.30,
    "club": 0.95,
    "major": 0.60,
}
# Most people share *something* with most people, so only each person's strongest
# few links are drawn. Node degree is deliberately left uneven: a well-connected
# person showing more links is real signal, and the graph is homogeneous in the
# sense that matters, which is that every node is the same kind of thing.
SHARED_LINKS_PER_PERSON = 3

EDGE_DASHES = {"observed_past": "solid", "observed_intent": "dash",
               "inferred": "dot"}
EDGE_WIDTHS = {"observed_past": 0.6, "observed_intent": 0.6, "inferred": 1.4}


def theme(name: str = DEFAULT_THEME) -> dict:
    return THEMES.get(name, DARK)


# Kept for callers and tests that read style by node kind. Colours resolve
# against the default theme; `theme()` is the source of truth at render time.
NODE_STYLE = {
    kind: dict(symbol=NODE_SYMBOLS[kind], size=NODE_SIZES[kind],
               colour=DARK["node"][kind])
    for kind in NODE_SYMBOLS
}
EDGE_TIERS = {
    tier: dict(dash=EDGE_DASHES[tier], width=EDGE_WIDTHS[tier],
               colour=DARK["edge"][tier])
    for tier in EDGE_DASHES
}
OBSERVED_PAST_EDGES = {"PARTICIPATES_IN", "STUDIES", "PLAYS_FOR",
                       "MEMBER_OF", "SHARES_WITH"}
OBSERVED_INTENT_EDGES = {"INTERESTED_IN"}

FOCUS_MODES = [
    # Default. Every person in place, so clusters stay full and stable and the
    # yellow short-list rings visibly move when a slider changes. Filtering to
    # the short list alone hid exactly the thing the sliders affect.
    "Cohort overview",
    "Short list",
    "Applicant focus",
    "Cohort isolate",
    # The one view where clubs are the subject rather than a characteristic, so
    # AC-22 to AC-24 still have somewhere to hold.
    "Clubs at succession risk",
    "Full population",
]

# ------------------------------------------------- NR-4 cluster geometry
# Ring radius is set so adjacent max-size discs stay clearly apart: with 12
# cohorts, adjacent centres are 2*R*sin(pi/12) apart, so R=2.80 leaves about
# 0.25 of clear space between two 0.60 discs. At R=2.45 the gap was 0.03 and the
# clusters read as touching.
# Floor only. Nothing occupies the centre now, so the ring is driven by disc
# packing rather than by reserving a clearing for attribute nodes.
CLUSTER_RING_RADIUS = 1.90
CLUSTER_DISC_MIN = 0.22
CLUSTER_DISC_MAX = 0.60         # radius scales with sqrt(count) between these
# Grouping by major produces far more clusters than grouping by cohort, so the
# ring has to grow or the discs collide. Radius is derived from the count rather
# than fixed: adjacent centres are 2*R*sin(pi/n) apart, and that must clear two
# max-size discs plus this margin.
CLUSTER_RING_MARGIN = 0.18
CLUSTER_CENTRE_CLEAR = 0.92     # attribute nodes stay inside this radius
GOLDEN_ANGLE = 137.50776405003785   # phyllotaxis, even fill without a seed
# within a cohort disc, applicants and alumni are offset in opposite directions
# so NR-4c holds by position as well as by shape
POPULATION_OFFSET = 0.30
# FR-8: one uniform tint for every cluster. A ramp would imply an ordering, and
# hue-per-cohort would compete with the node-kind palette NFR-3 depends on. The
# effect of a slider is reported in the cluster LABEL, as a count and a delta,
# rather than by brightening the disc, precisely so no cluster looks "better".
CLUSTER_FILL = DARK["cluster_fill"]
CLUSTER_LINE = DARK["cluster_line"]
CLUSTER_LABEL_COLOUR = DARK["label"]

# ------------------------------------------- NR-5 progressive disclosure
# The control is the explicit override (IR-8a). Low opacity is what makes the
# view zoom-responsive with no events at all: at wide view thousands of edges
# overlap in few pixels and read as a wash, and the same edges resolve into
# individual lines once panned and zoomed into a region.
DETAIL_LEVELS = {
    "Clusters only": dict(edges=False, edge_opacity=0.0, similar="never",
                          node_scale=0.70, labels=True),
    "Outlines": dict(edges=True, edge_opacity=0.06, similar="never",
                     node_scale=0.85, labels=True),
    "Connections": dict(edges=True, edge_opacity=0.35, similar="focus",
                        node_scale=1.00, labels=True),
    "Everything": dict(edges=True, edge_opacity=0.85, similar="always",
                       node_scale=1.00, labels=False),
}
DETAIL_ORDER = list(DETAIL_LEVELS)
DETAIL_DEFAULT = "Outlines"
# NR-5e: nothing may imply the graph is reacting to zoom when it is reacting to
# a setting. These are the only sanctioned descriptions.
DETAIL_DESCRIPTIONS = {
    "Clusters only": "Cohort clusters and labels. No connections drawn.",
    "Outlines": "Connections as faint texture. Pan and zoom in and they "
                "resolve into individual lines.",
    "Connections": "Connections drawn plainly, with the three kinds "
                   "distinguishable. Inferred similarity only for a selection.",
    "Everything": "Every connection including inferred similarity. Dense by "
                  "design.",
}

# ------------------------------------------------- caveats, FR-6
MOCK_DATA_CAVEATS = [
    ("Academic strength does not track advantage in this pool",
     "First-generation applicants average a 3.12 reader rating against 3.05 for "
     "continuing-generation, and public school 3.10 against private-independent "
     "2.97. The pool generator drew academic strength independently of income "
     "band, so the real-world correlation between socioeconomic advantage and "
     "measured credentials is absent. This dashboard therefore UNDERSTATES the "
     "Diversity against Academic Strength tension. Real data will show more."),
    ("Yield and academic strength show no correlation here",
     "Measured r = +0.005. Textbook expectation is inverse, since stronger "
     "applicants hold more offers. The generator does encode that, but merit "
     "awards and honours invitations to strong applicants offset it almost "
     "exactly. Do not present the absence of tension as a general finding."),
    ("Counting represented regions is useless as a diversity metric",
     "All four single-priority short lists span all seven regions at n=285. "
     "Diversity is therefore scored as distance from a target composition, "
     "not as a count of categories present."),
    ("Alumni outcomes are synthetic",
     "The entry-to-outcome relationship is a seeded construction with a "
     "Spearman correlation near 0.35, not a fitted model. Academic strength "
     "validated against this cohort demonstrates the mechanism, not a finding."),
]
