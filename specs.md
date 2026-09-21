# Interactive Admissions Similarity Graph

## Implementation Specification

The build contract. `requirements.md` states what and why, `design.md` states how
architecturally, this document states exactly which names, values, signatures, and
defaults to write. Declarative throughout; rationale lives in the other two.

---

## 0. Two Decisions That Belong Upstream

Both were forced by writing exact specifications and should be back-ported.

### 0.1 Affinity and religious clubs are a protected-attribute proxy

DR-6 makes club interest a similarity input via SR-7. A concrete club roster
contains affinity and religious organisations, and an applicant's interest in one
is a strong proxy for race or religion. Scoring similarity on shared affinity-club
interest reintroduces exactly what SR-3 excludes, through the back door.

**Specification.** `campus_clubs_dim.csv` gains a `proxy_sensitive` column beyond
the DR-6 column list. Clubs flagged `Y`:

| Included in | Flagged clubs |
| :--- | :--- |
| SR-8 succession projection, roster panels, graph rendering | **Yes.** A dying Hillel chapter is real information staff need |
| SR-7 club affinity similarity dimension | **No.** Excluded from `M[:,:,8]` entirely |
| SR-8c P3 club coverage scoring | **No.** Coverage credit would reward admitting by inferred identity |

The split is the point: you may know a chapter needs members without scoring an
applicant on inferred religion or race. Requires a DR-6 amendment.

### 0.2 Athletic team and Activity are the same node

DR-5 says model each sport once, inside DR-2. NR-1 lists Athletic team as its own
node type. Both hold: sports are rows in `activities_dim.csv` with
`category = Athletics`, and `graphbuild` types those nodes Athletic team for
rendering. There is one node per sport, not two.

Resulting exact node count, replacing design.md section 7.1's estimate of ~1,950:

| Type | Count |
| :--- | :--- |
| Applicant | 1,200 |
| Alum | 600 |
| Activity, non-athletic | 31 |
| Athletic team | 16 |
| Major | 33 |
| Campus club | 45 |
| **Total** | **1,925** |

---

## 1. Constants, `config.py`

```python
SEED          = 20260920
N_ALUM        = 600
N_DIM         = 9
CLASS_TARGET  = 285          # A6
COA           = 84_600       # matches aid model
EPS           = 1e-6

# IR-1a. Monotone. NEVER rendered in the interface.
WEIGHT = {"Not considered": 0.00, "Minor": 0.25, "Moderate": 0.50,
          "Major": 1.00, "Defining": 2.50}
SCALE = list(WEIGHT)         # widget options, order preserved

ROLE_MULT = {"Member": 1.0, "Officer": 1.3, "Captain": 1.4,
             "President": 1.5, "Founder": 1.6}
RECOG_MULT = {"None": 1.0, "School": 1.1, "Regional": 1.25,
              "State": 1.4, "National": 1.6}

TOPK_EDGES    = 3            # NR-3a default
TOPK_P4       = 10           # P4b neighbours
SIM_FLOOR     = 0.25         # NR-3b, and P4b thin-neighbourhood guard
BRIDGE_BOOST  = 1.4          # SR-7a, result capped at 1.0

STRENGTH_BANDS = [(0.90, "very strong match"), (0.75, "strong match"),
                  (0.50, "moderate match"), (0.25, "limited match"),
                  (0.00, "little in common")]
ROSTER_BANDS   = [("Secure", 1.25), ("Thin", 1.00),
                  ("At risk", 0.60), ("Critical", 0.00)]  # × min_viable
SHORTLIST_SIZE = {"Very selective": 0.60, "Selective": 1.00, "Broad": 1.60}
```

`SHORTLIST_SIZE` multiplies `CLASS_TARGET`, giving 171 / 285 / 456.

---

## 2. Activity Vocabulary, `activities_dim.csv`

47 rows. Columns: `activity_id`, `activity_name`, `category`.

| ID | Name | Category |
| :--- | :--- | :--- |
| ACT-01 | Debate and Speech | Competitive intellectual |
| ACT-02 | Model UN | Competitive intellectual |
| ACT-03 | Math Olympiad | Competitive intellectual |
| ACT-04 | Science Olympiad | Competitive intellectual |
| ACT-05 | Robotics | Competitive intellectual |
| ACT-06 | Quiz Bowl | Competitive intellectual |
| ACT-07 | Instrumental Music | Arts |
| ACT-08 | Vocal Music | Arts |
| ACT-09 | Theatre | Arts |
| ACT-10 | Dance | Arts |
| ACT-11 | Studio Art | Arts |
| ACT-12 | Film | Arts |
| ACT-13 | Newspaper | Publications |
| ACT-14 | Yearbook | Publications |
| ACT-15 | Literary Magazine | Publications |
| ACT-16 | Radio | Publications |
| ACT-17 | Student Government | Governance |
| ACT-18 | Class Officer | Governance |
| ACT-19 | Honour Council | Governance |
| ACT-20 | Community Volunteering | Service |
| ACT-21 | Tutoring and Peer Mentoring | Service |
| ACT-22 | Cultural and Affinity Organisations | Identity and culture |
| ACT-23 | International Student Association | Identity and culture |
| ACT-24 | Entrepreneurship | Enterprise |
| ACT-25 | Business and Investment | Enterprise |
| ACT-26 | DECA | Enterprise |
| ACT-27 | Independent Research | Research |
| ACT-28 | Published Work | Research |
| ACT-29 | Lab Assistantship | Research |
| ACT-30 | Part-time Employment | Obligation |
| ACT-31 | Family Caregiving | Obligation |
| ACT-32 … ACT-47 | Football, Field Hockey, Men's Soccer, Women's Soccer, Women's Lacrosse, Men's Lacrosse, Baseball, Softball, Swimming & Diving, Track & Field, Women's Basketball, Men's Basketball, Volleyball, Rowing, Golf, Wrestling | Athletics |

Sport names match `athletic_recruit_sport` in `mock_admit_pool_fall2026.csv`
exactly, so DR-1 athletics join without a mapping table.

---

## 3. Campus Clubs, `campus_clubs_dim.csv`

45 rows. Columns in order: `club_id`, `club_name`, `category`,
`linked_activity_id`, `current_members`, `graduating_members`,
`min_viable_members`, `charter_status`, `advisor_backed`, `proxy_sensitive`.

`R` marks the nine clubs required at succession risk by DR-6c, 20% of 45.
`P` marks `proxy_sensitive = Y`.

| ID | Club | Linked | Flags |
| :--- | :--- | :--- | :--- |
| CLB-01 | Debate Society | ACT-01 | |
| CLB-02 | Model United Nations | ACT-02 | |
| CLB-03 | Math Club | ACT-03 | |
| CLB-04 | Science Olympiad Team | ACT-04 | |
| CLB-05 | Robotics Team | ACT-05 | |
| CLB-06 | Quiz Bowl | ACT-06 | **R** |
| CLB-07 | Concert Band | ACT-07 | |
| CLB-08 | Orchestra | ACT-07 | |
| CLB-09 | Chamber Choir | ACT-08 | |
| CLB-10 | A Cappella Ensemble | ACT-08 | |
| CLB-11 | Student Theatre Company | ACT-09 | |
| CLB-12 | Dance Collective | ACT-10 | |
| CLB-13 | Studio Art Collective | ACT-11 | |
| CLB-14 | Film Society | ACT-12 | |
| CLB-15 | Student Newspaper | ACT-13 | |
| CLB-16 | Yearbook | ACT-14 | **R** |
| CLB-17 | Literary Review | ACT-15 | |
| CLB-18 | Campus Radio | ACT-16 | **R** |
| CLB-19 | Student Government Association | ACT-17 | |
| CLB-20 | Honour Council | ACT-19 | **R** |
| CLB-21 | Volunteer Corps | ACT-20 | |
| CLB-22 | Peer Tutoring Centre | ACT-21 | |
| CLB-23 | International Students Association | ACT-23 | |
| CLB-24 | Entrepreneurship Society | ACT-24 | |
| CLB-25 | Investment Club | ACT-25 | |
| CLB-26 | Undergraduate Research Collective | ACT-27 | |
| CLB-27 | Black Student Union | ACT-22 | **P** |
| CLB-28 | Latino Student Alliance | ACT-22 | **P** |
| CLB-29 | Asian Students Association | ACT-22 | **P** |
| CLB-30 | Hillel | ACT-22 | **P R** |
| CLB-31 | Newman Catholic Community | ACT-22 | **P** |
| CLB-32 | Muslim Students Association | ACT-22 | **P** |
| CLB-33 | Outdoors Club | | |
| CLB-34 | Chess Club | | **R** |
| CLB-35 | Esports Association | | |
| CLB-36 | Sustainability Coalition | | |
| CLB-37 | Habitat Build Chapter | ACT-20 | |
| CLB-38 | Pre-Health Society | | |
| CLB-39 | Mock Trial | ACT-01 | |
| CLB-40 | Political Union | ACT-17 | |
| CLB-41 | Photography Club | ACT-11 | |
| CLB-42 | Culinary Society | | **R** |
| CLB-43 | Board Game Guild | | **R** |
| CLB-44 | Ballroom Dance Society | ACT-10 | |
| CLB-45 | Astronomy Club | ACT-04 | **R** |

Seven of 45 carry no `linked_activity_id`: CLB-33, 34, 35, 36, 38, 42, 43.
Deliberate: a bridge that always exists carries no information, and campus-only
organisations are the realistic case. Six activities have no club counterpart
either — ACT-18, 26, 28, 29, 30, 31 — as do all 16 sports, which route through
`PLAYS_FOR` instead.

---

## 4. Edge Files

### 4.1 `activities_edges.csv`

Columns in order: `person_id`, `person_type`, `activity_id`, `context`, `role`,
`years_involved`, `recognition_level`.

| Parameter | Value |
| :--- | :--- |
| Edges per person | 3 to 6, mean 4.2 |
| Frequency distribution | Zipf, exponent **0.8**, over non-athletic activities. Amended from 1.6: at 1.6 the top activity is shared by 94% of people, which dominates the layout and carries no information. At 0.8 the spread runs 60% to 5%, preserving a ~6x idf range for SR-4 without a degenerate hub |
| `context` | `High School` for applicants; both values for alumni |
| `role` distribution | Member 0.62, Officer 0.20, Captain 0.08, President 0.06, Founder 0.04 |
| `recognition_level` | None 0.50, School 0.24, Regional 0.14, State 0.09, National 0.03 |
| Athletics edges | Only where DR-1 `athletic_recruit_sport` is non-null, or alum `varsity_seasons > 0` |

### 4.2 `club_edges.csv`

Columns in order: `person_id`, `person_type`, `club_id`, `relationship`,
`interest_signal`, `signal_strength`, `club_role`, `years_active`.

| Parameter | Value |
| :--- | :--- |
| Applicant `INTERESTED_IN` edges | 0 to 4, mean 1.8. 14% of applicants have none |
| Alum `MEMBER_OF` edges | 0 to 4, mean 2.1 |
| `interest_signal` | Portal Click 0.31, Info Session Topic 0.18, Campus Visit Meeting 0.12, Essay Mention 0.15, Application Checkbox 0.19, Counsellor Note 0.05 |
| `signal_strength` | Passive 0.46, Explicit 0.38, Sustained 0.16 |
| `club_role` | Member 0.68, Officer 0.19, President 0.08, Founder 0.05 |
| `years_active` | 1 to 4 |
| DR-6d association | Cramér's V ≈ 0.45 between an applicant's HS activities and interest in the linked club |

Null discipline, DR-6a: `INTERESTED_IN` rows leave `club_role` and `years_active`
empty; `MEMBER_OF` rows leave `interest_signal` and `signal_strength` empty. The
loader rejects any row violating this.

### 4.3 `alumni_outcomes.csv`

600 rows, cohorts 2019 through 2025. Columns exactly as DR-4, entry block then
outcome block, with **no giving field**, per DR-4b.

| Parameter | Value |
| :--- | :--- |
| `exemplar_flag = Y` | Top 200 by `success_composite_tier` |
| Each `success_*` component | Normalised [0,1], stored separately per DR-4c |
| `retained_year2` | 0.91 overall, correlated with entry rating |
| `graduated_4yr` | 0.78 |
| Entry-to-outcome signal | Spearman ≈ 0.35 between `entry_academic_rating` and `final_cum_gpa`. Non-zero so P4b carries signal; modest so it does not dominate |

---

## 5. Core API Contracts

Signatures are the contract. No `core/` module imports `streamlit`.

```python
# core/loader.py
def load(data_dir: Path) -> Dataset
def validate(ds: Dataset) -> None          # raises DataContractError

# core/features.py
def build_tensors(ds: Dataset) -> tuple[np.ndarray, np.ndarray]
    """Returns (M, A), each (n_app, n_alum, 9) float32."""
def dimension_names() -> list[str]         # length 9, index-aligned

# core/similarity.py
def contract(M2, A2, w: np.ndarray, shape) -> np.ndarray
def decompose(M, A, w, a_idx: int, e_idx: int) -> dict[str, float]
def topk(sim: np.ndarray, k: int, floor: float) -> np.ndarray

# core/priorities.py
def p1_yield(ds) -> np.ndarray
def p2_expected_ntr(ds) -> np.ndarray
def p4_academic(sim, ds, success_w, use_credential, use_validated
                ) -> tuple[np.ndarray, np.ndarray]   # (score, thin_mask)
def percentile(x: np.ndarray) -> np.ndarray
def build_shortlist(indiv_pct, C, target, B, at_risk, w_p3, w_club,
                    size: int, ids: np.ndarray) -> np.ndarray

# core/clubs.py
def club_affinity(ds) -> tuple[np.ndarray, np.ndarray]   # match, applicable
def project_rosters(ds) -> pd.DataFrame     # club_id, projected, band, at_risk
def sole_pipeline(ds, rosters) -> dict[str, list[str]]

# core/explain.py
def rationale(contribs, strength_pct, dropped, weakest) -> str
def attribute_change(prev_w, curr_w, prev_list, curr_list, recompute) -> dict
def counterfactual(applicant_idx, curr_w, recompute, size) -> str | None

# core/graphbuild.py
def build_nodes(ds) -> dict[str, Node]
def build_observed_edges(ds) -> list[Edge]
def freeze_layout(nodes) -> dict[str, tuple[float, float]]
def subgraph(mode: str, focus: str | None, ctx) -> tuple[list, list]
```

`p4_academic` returns `thin_mask` so the caller can honour design.md 5.5 without
inspecting internals.

---

## 6. Dimension Index

`M[:,:,d]` and `A[:,:,d]`, index-aligned with `dimension_names()`.

| d | Name | applicable = 0 when | Default |
| :-- | :--- | :--- | :--- |
| 0 | Academic preparation | never | Major |
| 1 | Field of study | never | Moderate |
| 2 | Activity portfolio | both sets empty | Major |
| 3 | Athletics | neither participated | Minor |
| 4 | Leadership depth | never | Moderate |
| 5 | School and geography | never | Minor |
| 6 | Socioeconomic context | never | **Not considered**, FR-2 |
| 7 | Engagement | never | Minor |
| 8 | Campus club affinity | both sets empty, after dropping `proxy_sensitive` clubs | Moderate |

---

## 7. Widget Inventory

Every importance control is `st.select_slider(options=SCALE)`. `st.slider` is
prohibited for weights, IR-1.

| Key | Label | Widget | Default |
| :--- | :--- | :--- | :--- |
| `w_yield` | Yield | select_slider | Moderate |
| `w_ntr` | Net tuition revenue | select_slider | Moderate |
| `w_div` | Diversity | select_slider | Moderate |
| `w_acad` | Academic strength | select_slider | Moderate |
| `shortlist_size` | Short list breadth | select_slider | Selective |
| `w_club_cov` | Club coverage within diversity | select_slider | Minor |
| `sim_dim_<0..8>` | per dimension | toggle + select_slider | section 6 |
| `succ_attainment` … `succ_contribution` | four success components | toggle | all on |
| `match_strictness` | Match strictness | select_slider | Moderate |
| `matches_per_applicant` | Matches shown | select_slider, `Closest only`/`A few`/`Several`/`Many` | A few |
| `show_clubs` | Show campus clubs | toggle | on |
| `at_risk_only` | Clubs at succession risk only | toggle | off |
| `show_sole_pipeline` | Surface sole-pipeline applicants | toggle | off |
| `focus_mode` | Focus | selectbox, NR-3d list | Short list |
| `preset` | Preset | selectbox | Balanced |
| `pins` | — | session_state list, max 4 | [] |
| `disclose_weights` | Show numeric weights | toggle | off, XR-3 |

`matches_per_applicant` maps to k of 1 / 3 / 6 / 12.
`match_strictness` maps to `SIM_FLOOR` of 0.10 / 0.18 / 0.25 / 0.35 / 0.50.

### 7.1 Presets, IR-5

| Preset | yield | ntr | div | acad |
| :--- | :--- | :--- | :--- | :--- |
| Balanced | Moderate | Moderate | Moderate | Moderate |
| Yield-first | Defining | Minor | Minor | Minor |
| Revenue-first | Minor | Defining | Not considered | Minor |
| Access and mission | Minor | Not considered | Defining | Minor |
| Academic-first | Minor | Minor | Minor | Defining |

The four single-priority presets are the AC-6 fixtures.

---

## 8. Rationale Templates, XR-2

```
"{strength}, driven mainly by {drivers}{despite}{dropped}."
```

| Slot | Construction |
| :--- | :--- |
| `strength` | `STRENGTH_BANDS` lookup on percentile |
| `drivers` | Dimension phrases, descending, until cumulative share ≥ 0.60, joined with "and" |
| `despite` | `", despite {phrase}"` for the lowest-scoring enabled dimension whose weight ≥ Moderate; omitted if none below 0.35 |
| `dropped` | `" (no {phrase} in common to compare)"` when any dimension was dropped under SR-7b |

Dimension phrases, index-aligned with section 6:

```
0 "a similar academic profile"        5 "a comparable school and region"
1 "the same intended field"           6 "a similar socioeconomic context"
2 "a shared {activity} background"    7 "comparable demonstrated interest"
3 "shared varsity athletics"          8 "shared interest in {club}"
4 "comparable leadership experience"
```

Slots 2 and 8 interpolate the highest-idf shared activity or club by name, which
is what makes a rationale specific rather than generic.

P4b thin-neighbourhood disclosure, design.md 5.5, is a fixed string:
`"Academic strength from credentials only; too few comparable alumni to validate."`

---

## 9. Rendering Contract

| Node type | `marker.symbol` | Size | Outline |
| :--- | :--- | :--- | :--- |
| Applicant | `circle` | 8 | filled when shortlisted |
| Alum | `square` | 9 | 3px when `exemplar_flag` |
| Activity | `diamond` | 6 + degree scaling | — |
| Athletic team | `triangle-up` | 10 | — |
| Major | `hexagon` | 11 | — |
| Campus club | `star` | by roster band | red 3px when at risk |

| Edge tier | Trace | `line.dash` |
| :--- | :--- | :--- |
| Observed past | `PARTICIPATES_IN`, `STUDIES`, `PLAYS_FOR`, `MEMBER_OF` | `solid` |
| Observed intent | `INTERESTED_IN` | `dash` |
| Inferred | `SIMILAR_TO` | `dot`, heavier width, distinct colour |

One Plotly trace per tier, `None`-separated coordinates. One trace per node type.
Six node traces plus three edge traces, nine total, regardless of graph size.

---

## 10. Task List

### Phase 1 — Data
- [ ] `config.py` with section 1 constants
- [ ] `activities_dim.csv`, 47 rows per section 2
- [ ] `campus_clubs_dim.csv`, 45 rows per section 3, including `proxy_sensitive`
- [ ] `generate_mock_graph_data.py`, seeded, emitting DR-2, DR-4, DR-6 per section 4
- [ ] `core/loader.py` with the five validation checks from design.md 3.1
- [ ] Schema `.md` per data file, matching project convention
- [ ] Verify DR-6c: at least nine clubs at risk, at least one with a sole pipeline

### Phase 2 — Scoring, headless
- [ ] `core/features.py`, nine dimensions per section 6
- [ ] `core/clubs.py` affinity, excluding `proxy_sensitive` from d=8
- [ ] `core/similarity.py` contract and decompose
- [ ] `core/priorities.py` P1, P2, P4, percentile
- [ ] `build_shortlist` with short-circuit and stable tie-breaks
- [ ] `core/clubs.py` roster projection and sole pipeline
- [ ] Tests: AC 6, 9, 10, 14, 18, 20, 26, 27 plus `test_no_streamlit_in_core`
- [ ] **Gate:** all Phase 2 tests green before any rendering work

### Phase 3 — Graph
- [ ] `core/graphbuild.py` nodes and observed edges, 1,925 nodes
- [ ] Deterministic layered `freeze_layout`, serialisable to `layout.json`
- [ ] Seven `subgraph` modes per design.md 7.3
- [ ] Verify coordinates are identical across two runs

### Phase 4 — Render, gated on spike
- [ ] Install `plotly`
- [ ] **Spike:** `st.plotly_chart(on_select="rerun")` fires on node and edge selection in Streamlit 1.61.1
- [ ] **Spike:** measure build, serialise, and paint at subgraph scale against the ~200 ms remaining after design.md section 12's Python subtotal
- [ ] `ui/graphview.py`, nine traces per section 9
- [ ] Decision point: proceed, or tighten `TOPK_EDGES` and `SIM_FLOOR`, or fall back to `st-link-analysis`

### Phase 5 — Explainability
- [ ] `core/explain.py` rationale per section 8
- [ ] Contribution bars, XR-1
- [ ] Change attribution and diff panel, XR-4
- [ ] Cost-of-choice table, XR-5
- [ ] Shared-path highlighting, XR-6
- [ ] Nearest exemplars, XR-7, and counterfactual, XR-8
- [ ] Configuration sentence, XR-9
- [ ] Export, XR-10
- [ ] Tests: AC 11, 13, 16

### Phase 6 — Clubs, fairness, caveats
- [ ] Clubs tab, IR-7d and IR-7e
- [ ] Composition panel and exemplar comparison, FR-1
- [ ] Drift warning, FR-3
- [ ] Suppression audit badge, FR-5
- [ ] Caveats tab carrying requirements 2.4, FR-6
- [ ] Tests: AC 17, 19, 22, 23, 24, 25
- [ ] Final sweep: all 27 acceptance criteria

---

## 11. Open Items

| Item | Blocks | Owner |
| :--- | :--- | :--- |
| DR-6 amendment for `proxy_sensitive`, section 0.1 | Phase 1 | Requirements |
| Whether campus vitality becomes a fifth priority, requirements SR-8 note | Phase 2 | Stakeholder |
| Plotly event and render viability, section 10 Phase 4 | Phase 4 | Spike |
| Whether to correct the mock pool's missing SES-to-academic correlation, requirements 2.4 | Phase 6 caveat wording | Stakeholder |
