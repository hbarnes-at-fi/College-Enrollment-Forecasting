# Interactive Admissions Similarity Graph

## Technical Design

Implements `requirements.md`. Every section traces to requirement IDs. Where a
design decision satisfies several requirements at once, or cannot satisfy one
without a tradeoff, that is stated rather than buried.

---

## 1. Design Summary

Five decisions carry the architecture.

| Decision | Serves | Why |
| :--- | :--- | :--- |
| Precompute a per-dimension match **tensor** once; interactions only re-weight it | NFR-1, SR-1 | Moves all pairwise work out of the interaction path. Measured 8.2 ms to re-weight |
| Carry a parallel **applicability mask** alongside the match tensor | SR-7b, IR-1b, SR-1b | One formula then handles weighted mean, zero-weight removal, and dropped dimensions |
| **Freeze the graph layout**, computed on observed edges only | NFR-2, XR-4 | Nothing moves on re-weight, so "what changed" is legible |
| Convert every priority to **percentile rank** before blending | IR-2, AC-6 | Makes probabilities, dollars, and similarities commensurable; monotone, so single-priority short lists are unchanged |
| **Short-circuit the greedy loop** when Diversity carries no weight | SR-5b, NFR-1 | Greedy is only required when P3 is active; otherwise a sort |

### 1.1 Verified, and not

Measured on this machine, numpy 2.5.1, at full stated scale of 1,200 applicants
against 600 alumni across 9 dimensions.

| Hot-path stage | Measured | Serves |
| :--- | :--- | :--- |
| Re-weight similarity, masked normaliser | 8.2 ms | SR-1 |
| Top-k `SIMILAR_TO` edge selection | 9.3 ms | NR-3a |
| P4 similarity-weighted kNN | 9.3 ms | 1.3(b) |
| Greedy 285 picks, composition plus club coverage | 40.4 ms | SR-5b, SR-8c |
| **Full interaction cycle** | **65.7 ms** | NFR-1 |
| Tensor build, cached, once at load | 20.9 ms | — |

Python compute fits NFR-1's 300 ms budget with roughly 4.5x headroom, and the
greedy loop is the dominant term at 40 of 66 ms.

**What this benchmark does not cover.** Streamlit rerun overhead, Plotly figure
construction, JSON serialisation, and browser paint. At full graph scale those
plausibly exceed the remaining 234 ms on their own. The budget therefore depends
on NR-3c holding: the default view must be a focused subgraph of a few hundred
nodes, not the full population. Compute was never the risk. Rendering is. See
section 12.

### 1.2 Dependencies

Present: `numpy` 2.5.1, `pandas` 3.0.3, `streamlit` 1.61.1.
**Missing and required: `plotly`.** Install before Phase 4.

`networkx` is deliberately **not** a dependency. Section 8 uses a deterministic
layered layout instead of a force-directed one, which is both more readable at
this scale and removes the need for it. Adjacency is plain dicts of sets.

---

## 2. Module Layout

```
streamlit_graph_demo/
  app.py                     entry point, Streamlit only
  config.py                  constants, weight maps, palettes, targets
  requirements.txt

  data/
    generate_mock_graph_data.py
    alumni_outcomes.csv          DR-4
    activities_dim.csv           DR-3
    activities_edges.csv         DR-2
    campus_clubs_dim.csv         DR-6
    club_edges.csv               DR-6
    *.schema.md                  project convention

  core/                      no Streamlit imports, ever
    loader.py                read and validate, DR-1 to DR-6
    features.py              build match tensor and mask, SR-2
    similarity.py            contraction and decomposition, SR-1
    priorities.py            P1-P4, greedy short list, SR-5
    clubs.py                 affinity and succession, SR-7, SR-8
    explain.py               rationale generation, XR-2
    graphbuild.py            nodes, edges, frozen layout, NR-1 to NR-3

  ui/
    graphview.py             Plotly figure assembly
    panels.py                detail, fairness, diff, caveat panels

  tests/
```

**Rule.** `core/` and `data/` must not import `streamlit`. Enforced by a test
that parses imports, not by convention. This is what makes scoring testable
headlessly, per requirements section 12.

---

## 3. Data Layer

### 3.1 Loading and validation, `core/loader.py`

```python
@dataclass(frozen=True)
class Dataset:
    applicants:  pd.DataFrame   # DR-1, 1200 rows
    alumni:      pd.DataFrame   # DR-4, ~600 rows
    activities:  pd.DataFrame   # DR-3 vocabulary
    act_edges:   pd.DataFrame   # DR-2
    clubs:       pd.DataFrame   # DR-6 dimension
    club_edges:  pd.DataFrame   # DR-6 edges
    app_index:   dict[str, int] # applicant_id -> row
    alum_index:  dict[str, int]

def load(data_dir: Path) -> Dataset
```

Validation is fail-loud at load, never silently coerced:

| Check | Requirement |
| :--- | :--- |
| Referential integrity on every FK in DR-2 and DR-6 | DR-2, DR-6 |
| `relationship` matches `person_type`: `INTERESTED_IN`/Applicant, `MEMBER_OF`/Alum | DR-6a |
| Applicant-only and alum-only club columns are null for the other kind | DR-6a |
| At least one club satisfies `current - graduating < min_viable` | DR-6c |
| `ethnicity_ipeds` and `gender` are dropped at load and absent from `Dataset` | SR-3, FR-5 |

**Design note on SR-3.** Protected attributes are removed in the loader rather
than merely unused downstream. A field that never enters the object cannot leak
into a score by later carelessness, and FR-5's audit becomes a one-line assertion
over `Dataset` columns instead of an inspection of every code path.

### 3.2 Generator, `data/generate_mock_graph_data.py`

Seeded, per NFR-5. Produces DR-2, DR-3, DR-4, DR-6 plus schema files.

Generation constraints that are requirements, not preferences:

- **Zipf activity frequency**, DR-2. Uniform frequency destroys SR-4's rarity weighting, since inverse frequency becomes constant.
- **Entry block correlated with outcome block**, DR-4. If entry attributes carry no signal about outcomes, P4(b) is noise and section 1.3 of the requirements collapses.
- **One fifth of clubs at genuine succession risk**, DR-6c.
- **Club interest correlated with, not determined by, high school activity** through `linked_activity_id`, DR-6d. Target roughly 0.45 association: strong enough that the bridge is real, weak enough that the club layer adds information.
- **Known gap left in place.** Requirements 2.4 records that academic strength does not track socioeconomic advantage in the existing pool, so the Diversity against Academic Strength tension under-displays. The generator does not silently patch this, because the applicant pool is fixed by A1 and inventing a correlation in the alumni file alone would produce an inconsistency between the two populations. FR-6 surfaces it in the interface instead.

---

## 4. Feature Precompute

### 4.1 The two tensors, `core/features.py`

```python
M: np.ndarray   # (n_app, n_alum, 9) float32, match score in [0,1]
A: np.ndarray   # (n_app, n_alum, 9) float32, applicability, 1.0 or 0.0
```

Flattened once to `(n_app * n_alum, 9)` for contraction. Memory 25.9 MB each,
51.8 MB total, measured. Built in 20.9 ms and cached.

`A` exists to serve SR-7b. Where a dimension is meaningless for a pair, two
applicants who both joined nothing, `A[i,e,d] = 0` and the dimension leaves that
pair's normaliser entirely. It is scored neither zero, which would falsely
penalise, nor one, which would falsely inflate.

### 4.2 The nine dimensions, SR-2

| # | Dimension | `match` | `applicable` when |
| :-- | :--- | :--- | :--- |
| 0 | Academic preparation | Mean of $1-\|r_a-r_e\|/4$ on reader rating and banded GPA distance | always |
| 1 | Field of study | Exact major 1.0; same school 0.5; else 0.0 | always |
| 2 | Activity portfolio | Rarity-weighted Jaccard, 4.3 | either set non-empty |
| 3 | Athletics | Tiered, 4.4 | either participated |
| 4 | Leadership depth | Banded distance on highest role rank and count | always |
| 5 | School and geography | Mean of region match and HS type match | always |
| 6 | Socioeconomic context | First-generation match and banded income distance | always, **off by default** per FR-2 |
| 7 | Engagement | Banded distance on demonstrated interest rating | always |
| 8 | Campus club affinity | Rarity-weighted Jaccard with SR-7a bridge boost, 6.1 | either set non-empty |

### 4.3 Activity portfolio, SR-4

Inverse document frequency over people, not over edges:

$$
\mathrm{idf}(x) = \ln\!\left(\frac{N_{\text{people}}}{\mathrm{df}(x)}\right)
\qquad
\mathrm{match}_2(a,e) = \frac{\sum_{x \in S_a \cap S_e} \mathrm{idf}(x)\, \rho(x)}
                             {\sum_{x \in S_a \cup S_e} \mathrm{idf}(x)\, \rho(x)}
$$

where $\rho(x)$ folds in DR-2 role and recognition:

```
role:        Member 1.0  Officer 1.3  Captain 1.4  President 1.5  Founder 1.6
recognition: None   1.0  School  1.1  Regional 1.25  State 1.4    National 1.6
```

Without the idf term every pair matches through Community Volunteering, which is
exactly the failure SR-4 names.

### 4.4 Athletics tiers

| Condition | match |
| :--- | :--- |
| Same sport, both captain | 1.00 |
| Same sport | 0.85 |
| Both varsity, different sport | 0.60 |
| One varsity only | 0.20 |
| Neither varsity | **not applicable**, dimension dropped |

The last row is SR-7b applied to athletics. Scoring mutual non-participation as
agreement would make every non-athlete a strong match for every other
non-athlete, inflating similarity across the largest segment of the pool.

---

## 5. Scoring

### 5.1 Contraction, `core/similarity.py`

The whole of SR-1 is one expression:

$$
\mathrm{sim}(a,e) \;=\; \frac{\sum_d w_d\, M[a,e,d]}{\max\!\left(\sum_d w_d\, A[a,e,d],\ \varepsilon\right)}
$$

```python
def contract(M2, A2, w, shape):
    num = M2 @ w
    den = A2 @ w
    return (num / np.maximum(den, 1e-6)).reshape(shape)
```

One formula, four requirements:

| Requirement | Satisfied by |
| :--- | :--- |
| SR-1 additive weighted mean | numerator is linear in `w` |
| SR-1b comparability as weights change | division by `A2 @ w` |
| IR-1b `Not considered` truly removes | `w_d = 0` cancels from numerator and denominator together |
| SR-7b inapplicable dimensions dropped | `A[a,e,d] = 0` removes only that pair's term |

Decomposition for XR-1:

$$
\mathrm{contrib}_d(a,e) = \frac{w_d\, M[a,e,d]}{\sum_{d'} w_{d'} A[a,e,d']}
\qquad
\sum_d \mathrm{contrib}_d = \mathrm{sim}(a,e)
$$

Exact by construction, not by tolerance. Tested as a property, AC-10.

### 5.2 Word scale to weight, IR-1a

```python
# config.py — documented here, NEVER rendered in the interface
WEIGHT = {
    "Not considered": 0.00,
    "Minor":          0.25,
    "Moderate":       0.50,
    "Major":          1.00,
    "Defining":       2.50,
}
```

Monotone increasing, per IR-1a. The jump to 2.50 is deliberate: `Defining` has to
actually dominate. Against three priorities at `Moderate` it takes 62% of total
weight, and against three at `Not considered` it takes 100%, which is what AC-6
requires in order to reproduce the requirements 2.2 table.

### 5.3 Making four priorities commensurable

P1 is a probability, P2 is dollars, P4 is a similarity, P3 is a distance
reduction. Blending the raw values would let the dollar scale swamp everything.

Each of P1, P2, P4 is converted to **percentile rank within the pool** before
blending. Percentile rank is a monotone transform, so the ordering induced by any
single priority is unchanged, and AC-6 still reproduces the 2.2 table exactly.

P3 cannot be pre-percentiled because it is set-dependent, per SR-5. Inside the
greedy loop its marginal gain is normalised against the best available gain among
remaining candidates at that step, keeping it on the same [0,1] footing.

### 5.4 The four priorities, `core/priorities.py`

| | Source | Shape |
| :--- | :--- | :--- |
| **P1** Yield | `predicted_yield_prob` | individual, precomputed |
| **P2** Net tuition revenue | `net_price` × `predicted_yield_prob`, SR-6 | individual, precomputed |
| **P3** Diversity | marginal composition and club coverage gain | **set-dependent**, SR-5 |
| **P4** Academic strength | 5.5 | individual, depends on similarity weights |

P1 and P2 share `predicted_yield_prob` and are therefore not independent. SR-6
requires this be stated at the controls, not just in the source.

### 5.5 P4, alumni-validated academic strength

Requirements 1.3 specifies outcome-grounded rather than credential-based, with
credentials visible separately. Two components, independently toggleable:

**P4a credential.** Percentile of reader rating and recalculated GPA. The
conventional measure, retained for comparison.

**P4b alumni-validated.** Similarity-weighted kNN regression on the success label:

$$
P4b(a) \;=\; \frac{\sum_{e \in \mathrm{top}k(a)} \mathrm{sim}(a,e)\cdot \mathrm{success}(e)}
                   {\sum_{e \in \mathrm{top}k(a)} \mathrm{sim}(a,e)}
$$

`success(e)` is the weighted mean of the four DR-4 components that are currently
toggled on, per requirements 1.4, each normalised to [0,1]:

```
success_attainment · success_growth · success_completion · success_contribution
```

kNN is chosen precisely because it is nameable. The answer to "why does this
applicant score well" is a list of alumni, which SR-1a demands and an embedding
cannot provide.

**Thin-neighbourhood guard.** If every one of an applicant's top-k alumni falls
below a similarity floor, P4b rests on no real evidence. In that case P4b is
withheld, P4 falls back to P4a alone, and the rationale says so explicitly. This
is the common case for an applicant in a rare major with no alumni analogue, and
producing a confident score from thin evidence would be the most dangerous
failure mode in the tool.

### 5.6 Greedy short list, SR-5b and SR-8c

```python
def build_shortlist(indiv_pct, C, target, B, at_risk, w_p3, w_club, size, ids):
    if w_p3 == 0.0:
        return stable_top_n(indiv_pct, size, ids)      # SR-5b not engaged

    chosen, counts, covered = set(), zeros(n_cats), zeros(n_clubs)
    for k in range(size):
        d_new      = abs((counts + C) / (k + 1) - target).sum(axis=1)
        gain_comp  = -d_new
        gain_club  = (B * at_risk * (1 - covered)).sum(axis=1)
        gain_p3    = unit_normalise(gain_comp + w_club * gain_club)
        score      = indiv_pct + w_p3 * gain_p3
        score[already_chosen] = -inf
        pick = stable_argmax(score, ids)
        chosen.add(pick); counts += C[pick]; covered = maximum(covered, B[pick])
    return chosen
```

Four design points:

- **Short-circuit.** When P3 is `Not considered` the loop is skipped entirely. This is both the NFR-1 optimisation and the reason AC-6 reproduces the other three priorities exactly rather than approximately.
- **Vectorised inner step.** `counts + C` broadcasts over all 1,200 candidates at once. The loop is over the 285 picks, never over candidates. This is what holds the measured 40.4 ms.
- **Stable tie-breaks.** `np.argmax` resolves ties by array position, which makes output depend on CSV row order. Ties break on `applicant_id` ascending instead, per NFR-5.
- **Greedy is not optimal**, and the requirements already concede this in AC-6's "within greedy-selection tolerance." Each pick is locally best given prior picks. This is inherent, must be stated in the interface, and is the correct tradeoff against an exact combinatorial solve that cannot run in 300 ms.

---

## 6. Club Layer, `core/clubs.py`

### 6.1 Affinity, SR-7

Applicant `INTERESTED_IN` set against alum `MEMBER_OF` set, rarity-weighted as in
4.3. SR-7a adds a bridge boost: interest in a club whose `linked_activity_id` the
applicant actually pursued in high school is multiplied by 1.4, capped at 1.0.
Four years of Debate behind a stated interest in Debate Society is stronger
evidence than a portal click, and the bridge field is what lets the model see it.

Both sets empty means the dimension is dropped, per SR-7b and 4.1.

### 6.2 Succession projection, SR-8

$$
\mathrm{projected}(c) = \mathrm{current}(c) - \mathrm{graduating}(c)
  + \sum_{a \,\in\, \mathrm{interested}(c)} \mathrm{yield}(a)
$$

At risk when `projected < min_viable_members`, per SR-8a.

**SR-8d display rule.** The sum is probability-weighted, so it is an expectation
and not a headcount. It renders as a qualitative band, `Secure` · `Thin` ·
`At risk` · `Critical`, never as a decimal. This also keeps the club panel
consistent with IR-1's no-numbers philosophy.

### 6.3 Sole-pipeline applicants, SR-8b

For each at-risk club, rank interested applicants by yield probability and flag
those who are the only, or one of at most two, realistic sources of membership.

These applicants typically rank nowhere on P1, P2, or P4. IR-7e surfaces them
outside the composite ranking, and it is the one path in the interface that
deliberately promotes low-ranked applicants. It must be labelled as such, per
IR-7e, because an unlabelled override is indistinguishable from a bug.

---

## 7. Graph Construction, `core/graphbuild.py`

### 7.1 Nodes and adjacency

Node types per NR-1: Applicant, Alum, Activity, Major, Athletic team, Campus
club. Approximate counts: 1,200 + 600 + 60 + 25 + 16 + 45 ≈ 1,950.

Adjacency as `dict[str, set[str]]`. The operations needed are neighbours and
k-hop subgraph extraction, both trivial on dicts, which is why networkx is not a
dependency.

### 7.2 Frozen layout, NFR-2

The single most important rendering decision.

**Layout is computed once, over observed edges only, and never recomputed.**
`SIMILAR_TO` edges are excluded from layout entirely because they change with
every weight adjustment; including them would move every node on every
interaction and defeat XR-4, which depends on the user being able to see what
moved.

Two arrangements, both deterministic, both frozen. NR-4 makes the clustered one
the default; the arc one is retained because IR-8d requires clustering be
removable.

**Arrangement A, cohort clusters (default, NR-4).**

```
                     Recruited athletes
        Highly engaged                 Performing arts
    Access & first-gen    activities   Pre-law & civic
    International          majors      Research sciences
        Top academic       teams·clubs      Engineering
                  Business      Health sciences
```

Twelve cohort discs on a ring of radius 2.4, attribute nodes in the central
region inside radius 0.9. Within each disc, applicants fill a sub-disc offset one
way and alumni the other, so NR-4c holds by position as well as by shape.

Three properties worth stating because they are load-bearing:

- **Disc radius scales with √count.** Cohorts range from roughly 90 to 250 people. Fixed-radius discs would make the large ones unreadably dense and the small ones look insignificant, which would imply a ranking and breach FR-8.
- **Members are placed by phyllotaxis**, `r = R√((i+½)/n)`, `θ = i·137.508°`, ordered by id. Even fill, no clumping, fully deterministic, no random seed.
- **A ring has no beginning.** Cluster order around the ring therefore encodes no rank, which is the cheapest available way to satisfy FR-8. A left-to-right row would have implied one.

**Arrangement B, population arcs (fallback, IR-8d).**

```
        activities · majors · teams · clubs          central bands, by category
    applicants (left arc)              alumni (right arc)
```

The original arrangement: attribute nodes in central horizontal bands, applicants
and alumni on opposing arcs ordered by dominant attribute. It shows connectivity
well and structure poorly, which is exactly why NR-4 exists. Retained under
`IR-8d` because clustering imposes one reading of the pool and a user must be able
to remove it.

Both are cached in `st.cache_resource` and serialised to `layout.json` under
separate keys, since switching arrangement must not trigger a recompute.

### 7.2a Why not force-directed or community detection

A spring layout at 1,925 nodes produces an undifferentiated hairball and is the
slowest thing in the build. More importantly, so would any *discovered* clustering:
community detection would find real structure and label none of it, and an
unnamed cluster cannot be explained. That directly contradicts SR-1a, which
forbids methods whose output a committee cannot interrogate. NR-4b therefore
specifies named rule-based cohorts, and this is the same argument that ruled out
embeddings for scoring in 5.1.

### 7.2b Cohort assignment, `core/cohorts.py`

Both populations are assigned by the same predicates, applicants on their current
record and alumni on their `entry_*` block (NR-4a). To make one predicate serve
two schemas, each person is first normalised to a `Profile`:

```python
@dataclass(frozen=True)
class Profile:
    pid: str
    population: str        # "Applicant" | "Alum"
    sport: str
    school: str
    major: str
    rating: int
    gpa: float
    first_gen: bool
    income: str
    international: bool
    interest: int
    activities: frozenset[str]   # activity_id
```

Predicates then read one shape. Cohorts 3 and 4 are conjunctions of a major and an
activity, which is why `activities` is on the Profile at all.

```python
def assign(ds) -> Assignment      # primary cohort + all memberships, both populations
def definitions() -> list[CohortDef]   # for XR-12 disclosure
```

Three rules that are easy to get wrong:

- **Precedence, not first match on an arbitrary order.** A person matches several cohorts; the primary is the earliest in the published precedence (NR-4d). Smaller and operationally distinct cohorts come first, so a recruited athlete with a 3.95 GPA lands in Recruited athletes rather than Top academic.
- **All memberships are retained**, not just the primary, so selection can show the full set. Only position is single-valued.
- **Undersized cohorts merge into General** (NR-4e), computed over the two populations jointly. A cohort with four applicants and no alumni is noise, and it also breaks the NR-4a comparison the clustering exists to enable.

**FR-7 is enforced structurally, not by review.** `Profile` has no ethnicity or sex
field, because `Dataset` has none — they were dropped at load per SR-3. A cohort
predicate therefore cannot reference them even by mistake. `Profile.activities`
excludes club membership entirely, so `proxy_sensitive` clubs cannot leak in
either. The guardrail is that the data a predicate can see does not contain the
forbidden thing.

### 7.3 Subgraph extraction, NR-3

| Mode | Contents |
| :--- | :--- |
| Applicant focus | applicant, its top-k alumni, all shared attribute nodes |
| Activity focus | activity and every person attached |
| Major cohort | major, its applicants and alumni |
| **Club focus** | club, member alumni with outcomes, interested applicants, IR-7d |
| **At-risk clubs** | every club below viability and its pipeline, IR-7b |
| Short list | short list members and their evidence subgraphs, IR-3b, IR-3c |
| **Cohort isolate** | one cohort, both populations, every other cluster dimmed, IR-8c |
| Full population | explicit request only, with a warning, NR-3c |

`SIMILAR_TO` edges are capped at k per applicant, default 3, NR-3a, and
suppressed below a relevance floor even when under k, NR-3b.

---

## 8. Rendering, `ui/graphview.py`

### 8.1 Plotly trace structure

Performance depends on one thing: **one trace per edge tier, not one trace per
edge.** Coordinates are concatenated with `None` separators so an entire tier is
a single trace.

```python
edge_x = [x0, x1, None, x2, x3, None, ...]
```

Traces, matching NR-2a's three tiers:

| Trace | Edges | Style |
| :--- | :--- | :--- |
| Observed past | `PARTICIPATES_IN`, `STUDIES`, `PLAYS_FOR`, `MEMBER_OF` | solid |
| Observed intent | `INTERESTED_IN` | dashed |
| Inferred | `SIMILAR_TO` | distinct colour and width |

Plus one scatter trace per node type.

### 8.2 Shape and colour, NFR-3

Shape carries node type independently of colour, since colour alone fails
colour-blind users:

| Node | Symbol | Encoding |
| :--- | :--- | :--- |
| Applicant | circle | fill by short list membership |
| Alum | square | heavier outline when `exemplar_flag` |
| Activity | diamond | size by degree |
| Major | hexagon | — |
| Athletic team | triangle-up | — |
| **Campus club** | **star** | size by roster health, red outline at succession risk, NR-1, SR-8a |

### 8.3 Events, and what the component will not give us

`st.plotly_chart(fig, on_select="rerun", selection_mode=("points",))` on
Streamlit 1.61.1. Parameters verified present; that events fire on a node click
needs a browser and remains unproven.

**Measured limit.** `selection_mode` accepts only `points`, `box` and `lasso`.
There is no relayout hook, so the server never learns the current pan or zoom
state. This is why NR-5 cannot be implemented as continuous zoom-driven
level-of-detail, and the constraint belongs to the component rather than to the
layout choice. A custom JavaScript component would lift it, at a scope cost that
has not been agreed.

Fallback if selection proves broken: `st-link-analysis`, cytoscape-backed, richer
event model, additional dependency.

### 8.4 Progressive disclosure, NR-5

Four discrete levels, set by the word-scaled control in IR-8a. The control is
honest about being a setting: nothing in its label claims the graph is reacting to
zoom (NR-5e).

| Level | Observed edges | `SIMILAR_TO` | Node scale | Cohort labels |
| :--- | :--- | :--- | :--- | :--- |
| `Clusters only` | none | never | 0.70 | on |
| `Outlines` | opacity 0.06, texture | never | 0.85 | on |
| `Connections` | opacity 0.35, tiers legible | on focus only | 1.00 | on |
| `Everything` | opacity 0.85 | always | 1.00 | off |

Two mechanisms are doing the work and they are worth separating.

**The control is the explicit override.** It sets what gets drawn.

**Opacity is what makes the view zoom-responsive without any events at all.** At
`Outlines`, six thousand edges at 0.06 alpha overlap in few pixels and read as a
faint wash, so the clusters dominate. Pan and zoom into a region and the same
edges spread over more screen area, each one now alone against the background at
its own alpha, and they resolve into individual lines. No server round trip, no
relayout event, no state. This is the mechanism that actually delivers the
requested behaviour; the control exists so the user is not dependent on it.

Three rules on top:

- **NR-5c**, `SIMILAR_TO` never appears in the two widest levels, and at `Connections` only when a focus or selection is active. A wide view has no business asserting inference.
- **NR-5d**, XR-6 shared-path highlighting **overrides the level entirely**. If a user asks why two people are connected, the answer is drawn at any detail setting, including `Clusters only`. Implemented by appending the highlight traces after the level filter rather than inside it.
- **XR-13**, the active level appears in the XR-9 configuration sentence, so a view with no visible edges is never ambiguous between "no edges exist" and "edges are not being drawn".

### 8.5 Cluster chrome

Cohort discs render as twelve Plotly layout shapes (light uniform fill) plus
twelve annotations for the labels. Shapes and annotations are layout-level, not
traces, so the nine-trace contract in specs §9 is unaffected and the cost is
constant.

Fill is a single neutral tint for every cluster. A colour ramp would imply an
ordering and breach FR-8, and hue-per-cohort would compete with the node-kind
palette that NFR-3 depends on.

---

## 9. State Management

Streamlit reruns top to bottom, so cache scope is a correctness concern and not
only a performance one.

| Object | Mechanism | Key |
| :--- | :--- | :--- |
| CSV frames | `@st.cache_data` | data dir and file mtimes |
| Activity and club idf vectors | `@st.cache_data` | dataset hash |
| Composition one-hot `C`, club matrix `B` | `@st.cache_data` | dataset hash |
| Match tensor `M`, mask `A` | `@st.cache_resource` | dataset hash |
| Frozen layout | `@st.cache_resource` | dataset hash |
| Control values, pins, preset | `st.session_state` | — |
| Previous short list and previous weights | `st.session_state` | XR-4 |

**Cache hygiene.** `M` and `A` must never be keyed on weights. Build is cached,
contraction is not. Confusing the two would reintroduce the 20.9 ms build into
every interaction and, worse, silently return stale similarity.

**XR-11.** No cross-session persistence. `cache_resource` is process-scoped,
`session_state` is not written to disk, and `layout.json` holds only geometry.

---

## 10. Explainability, `core/explain.py`

### 10.1 Rationale generation, XR-2

```python
def rationale(contribs: dict[str, float],
              strength_pct: float,
              dropped: list[str],
              weakest: tuple[str, float] | None) -> str
```

1. Sort contributions descending.
2. Accumulate until cumulative share reaches 60%. Those are the named drivers.
3. Map each dimension to a phrase fragment.
4. Prefix a strength adverb from the band table below.
5. Append a `despite` clause naming the weakest enabled dimension carrying real weight.
6. Append a note for any dimension dropped under SR-7b.

Step 5 is what makes rationales read as honest rather than promotional. Step 6
prevents a silent absence of evidence from looking like agreement.

Strength bands, words only, per XR-3:

| Percentile | Word |
| :--- | :--- |
| ≥ 0.90 | very strong match |
| ≥ 0.75 | strong match |
| ≥ 0.50 | moderate match |
| ≥ 0.25 | limited match |
| below | little in common |

### 10.2 Change attribution, XR-4

Previous short list and previous weight vector live in `session_state`. For each
applicant that entered or left, revert one priority at a time to its prior value
and recompute. The single revert that flips membership is the attributed cause; if
no single revert flips it, report a combined effect.

Cost is four extra contractions, roughly 33 ms, which fits inside the measured
budget.

### 10.3 Cost of the choice, XR-5

When any priority sits at `Defining`, recompute the requirements 2.2 table for
the current short list beside the whole-pool baseline. Cheap aggregation over
1,200 rows.

XR-5 is the requirement that keeps the tool advisory rather than persuasive.
Maxing net revenue must visibly report that low-income representation went to
zero, which the requirements measured at exactly 0% against an 18% pool baseline.

### 10.4 Counterfactual, XR-8

For the selected applicant only, find the smallest single word-scale step on any
one priority that would bring them into the short list. Twenty evaluations at
most, four priorities by five levels, computed on demand rather than for the whole
pool.

### 10.5 Configuration sentence, XR-9

Rendered from the control state on every pass, always on screen. Word-scale
labels appear verbatim, so the sentence never contains a number.

---

## 11. Interface Layout

**Sidebar.** Preset selector, IR-5. Four priority controls, IR-2. Short list size,
IR-3a. P3 sub-controls: composition targets, SR-5c, and club coverage weight,
IR-7c. Similarity dimension toggles, SR-2, with socioeconomic off by default,
FR-2. Club show or hide and at-risk filter, IR-7a, IR-7b. Reset, IR-6.

**Header strip.** Configuration sentence, XR-9. Suppression audit badge, FR-5.

**Tabs.**

| Tab | Contents |
| :--- | :--- |
| Graph | Plotly view, focus selector, selected-node detail with contribution bars and rationale |
| Short list | Ranked table, evidence subgraph per row, export, XR-10 |
| Why | Full decomposition for the selected pair, XR-1, nearest exemplars, XR-7, counterfactual, XR-8 |
| Clubs | Roster projections, at-risk clubs, sole-pipeline applicants, IR-7d, IR-7e |
| Composition | Class projection against targets, exemplar cohort beside applicant pool, FR-1, drift warning, FR-3 |
| What changed | Entered, left, attributed cause, XR-4, and cost of the choice, XR-5 |
| Caveats | Requirements 2.4 limitations surfaced in-app, FR-6 |

No element is named or styled as a decision, per FR-4.

---

## 12. Performance Budget

| Stage | Measured or estimated | Note |
| :--- | :--- | :--- |
| Contraction | 8.2 ms measured | |
| Top-k | 9.3 ms measured | |
| P4 kNN | 9.3 ms measured | |
| Greedy short list | 40.4 ms measured | dominant compute term |
| XR-4 attribution | ~33 ms estimated | four extra contractions |
| **Python subtotal** | **~100 ms** | |
| Streamlit rerun overhead | not measured | |
| Plotly build, serialise, paint | **not measured, the real risk** | |
| **Budget, NFR-1** | **300 ms** | |

**The honest position.** Compute has roughly 3x headroom after including XR-4.
The unmeasured render path is what could break NFR-1, and it scales with rendered
node and edge count rather than with population. Mitigation is already a
requirement: NR-3c mandates a focused subgraph by default, a few hundred nodes
rather than 1,950. If the Phase 4 spike shows the render path exceeding budget at
subgraph scale, the correct response is to tighten NR-3a's default k and the NR-3b
relevance floor, not to weaken NFR-1.

---

## 13. Testing

Property tests, mapped to acceptance criteria.

| Test | Asserts | AC |
| :--- | :--- | :--- |
| `test_contributions_sum` | decomposition sums to score, exact | 10 |
| `test_not_considered_is_inert` | zero weight changes no score anywhere | 9 |
| `test_defining_reproduces_table` | one priority at `Defining` reproduces requirements 2.2 | 6 |
| `test_percentile_monotone` | percentile transform preserves single-priority order | 6 |
| `test_greedy_differs_from_sort` | greedy P3 output differs from a static diversity sort | 14 |
| `test_determinism` | identical config yields identical short list, twice | 20 |
| `test_no_protected_attributes` | `ethnicity_ipeds`, `gender` absent from `Dataset` and every feature name | 18 |
| `test_no_streamlit_in_core` | import parse over `core/` and `data/` finds no streamlit | section 12 |
| `test_empty_sets_dropped` | pairs sharing no clubs and no sports have those dims dropped, not scored | 26 |
| `test_club_risk_present` | generated data contains at least one at-risk club with a sole pipeline | 25 |
| `test_club_inert_when_p3_off` | club coverage changes nothing when P3 is `Not considered` | 27 |
| `test_thin_neighbourhood_fallback` | P4 falls back to P4a and says so when neighbours are below floor | 5.5 |
| `test_rationale_mentions_dropped` | rationale names dropped dimensions | XR-2 |
| `test_edge_tiers_distinct` | the three NR-2a tiers carry distinct style keys | 23 |
| `test_every_person_has_one_primary_cohort` | assignment is total and single-valued | 30 |
| `test_cohort_precedence_is_published_order` | an athlete with a 3.95 GPA lands in Recruited athletes | 30 |
| `test_both_populations_in_every_cohort` | no cohort is applicant-only or alum-only after merging | 29, 31 |
| `test_no_cohort_below_minimum` | undersized cohorts merged into General | 31 |
| `test_cohort_predicates_cannot_see_protected_fields` | `Profile` has no ethnicity, sex or club membership | 39 |
| `test_detail_levels_gate_edges` | `Clusters only` emits zero edge traces; `Everything` emits all | 32, 33 |
| `test_highlight_overrides_detail_level` | shared-path traces present even at `Clusters only` | 34 |
| `test_cluster_layout_deterministic` | identical positions across two builds, both arrangements | 41 |
| `test_clustering_is_removable` | arrangement B reachable and distinct | 36 |
| `test_cluster_fill_encodes_no_ranking` | one uniform tint, no per-cohort ramp | 40 |

---

## 14. Risks

| Risk | Severity | Response |
| :--- | :--- | :--- |
| Plotly render and transport exceed the remaining budget | Resolved | Measured 40 ms worst non-full mode after switching to numpy coordinate arrays. Plotly validates Python lists element by element; numpy skips that path |
| `on_select` event firing on node clicks | Medium, open | Parameters verified present, behaviour unproven without a browser |
| Zoom-driven level-of-detail is not achievable in the component | Accepted, designed around | 8.4. Opacity gives the behaviour without events; the IR-8a control gives explicit override. Neither is promised as zoom |
| Cohort predicates drift from the disclosed definitions | Medium | XR-12 renders definitions from the same `CohortDef` objects the predicates use, so the disclosure cannot disagree with the behaviour |
| Cluster sizes too uneven to read | Low | √count disc radius, and NR-4e merges undersized cohorts into General |
| Mock data lacks the SES to academic correlation, requirements 2.4 | Medium | Not patched, see 3.2. Surfaced in-app per FR-6 |
| Thin kNN neighbourhoods in rare majors | Medium | Guard in 5.5, fallback plus disclosure |
| Greedy short list is not globally optimal | Low | Inherent. Stated in-interface and conceded by AC-6 |
| `plotly` not installed | Low | Install step, Phase 4 |

---

## 15. Build Order

| Phase | Delivers | Exit condition |
| :--- | :--- | :--- |
| 1 | Generator, DR-2, DR-3, DR-4, DR-6, schema files | Loader validation passes, DR-6c satisfied |
| 2 | `core/` scoring, headless, with tests | AC 6, 9, 10, 14, 18, 20, 26, 27 green |
| 3 | Graph build and frozen layout | Deterministic coordinates, subgraph modes work |
| 4 | **Plotly spike**, then render and events | Selection events fire; render inside budget at subgraph scale |
| 5 | Explainability panels, XR-1 to XR-11 | AC 11, 13, 16 green |
| 6 | Clubs, composition, and caveat panels | AC 22 to 25, 17, 19 green |

Phase 2 completes before any rendering work, since every acceptance criterion
about correctness is testable headlessly and none of them needs a browser.
Phase 4 is gated on a spike because it carries the only high-severity risk.
