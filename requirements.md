# Interactive Admissions Similarity Graph

## Requirements Specification

A Streamlit dashboard that renders two populations as one graph, the current
admit pool and prior students who succeeded here, links them through shared
activities, majors, athletics, and campus clubs, and scores applicant-to-alum
similarity. The wide view is organised into named cohort clusters with the edge
layer suppressed to texture; detail resolves as the view narrows (NR-4, NR-5).
Four institutional priorities are reweighted with word-labelled controls.
Maxing any one priority selects a short list and highlights the evidence behind
each pick. Explainability outranks accuracy. The tool is advisory only.

---

## 1. What a Successful Candidate Looks Like

### 1.1 Two layers, deliberately separated

"Who we want" mixes two questions that must not be blended into one number:

| Layer | Question | Measured on |
| :--- | :--- | :--- |
| **Institutional priority** | Does admitting this person advance our four goals? | The applicant record, available today |
| **Alumni-validated promise** | Do students with this profile actually thrive here? | The alumni cohort, through the graph |

Three of the four priorities are computable from the applicant record alone. The
alumni graph earns its place on the fourth, Academic Strength, and that is where
the defensible definition lives.

### 1.2 The four priorities

**P1 Yield** — probability the applicant enrols. Individual score. Source:
`predicted_yield_prob`.

**P2 Net Tuition Revenue** — **expected** net revenue, `net_price` weighted by
probability of enrolling. Sticker net price alone is the wrong measure: a full-pay
applicant who will not enroll contributes nothing. This must be expected NTR.

**P3 Diversity** — marginal contribution to class composition targets across
**non-racial** dimensions: geographic region, socioeconomic band,
first-generation status, high school type, urban and rural context, program and
major mix, international representation, and **campus club coverage** per SR-8c.
Race and sex are excluded everywhere, per SR-3.

**P4 Academic Strength** — see 1.3. Not raw credentials.

### 1.3 Academic Strength, the defensible definition

Two ways to measure it, and the choice is the substance of the question:

| | Measures | Problem |
| :--- | :--- | :--- |
| (a) Entering credentials | GPA, rigour, test scores, reader rating | Measures preparation, which tracks opportunity. An input proxy standing in for an outcome. |
| (b) Alumni-validated promise | Similarity to alumni who thrived here | Requires the alumni cohort and careful bias instrumentation |

**Specify (b), with (a) visible as a separate component.** The argument:

1. **Outcome-grounded.** It asks whether students with this profile succeeded here, rather than whether the profile looks impressive.
2. **Contextual.** A 3.6 from a school whose alumni thrive is stronger evidence than a 3.9 from a school whose alumni struggle. Credentials cannot express this; the graph can.
3. **Auditable.** The score decomposes into named alumni and shared attributes. A reader can ask "which students is this based on" and get an answer. A credential index cannot be interrogated that way.
4. **Less advantage-correlated.** Raw credentials partly encode access to test prep, AP offerings, and counselling. Outcome-grounding attenuates this rather than laundering it.

### 1.4 The alumni success label

Resolves the earlier open question. A composite of four components, each
independently toggleable so staff can see which definition drives a match:

| Component | Fields | Captures |
| :--- | :--- | :--- |
| **Attainment** | `final_cum_gpa` | Classroom performance |
| **Growth** | `gpa_delta`, first year to final | Adaptation, rewards students who start behind and climb |
| **Completion** | `retained_year2`, `graduated_4yr`, `academic_probation_ever` | Follow-through |
| **Contribution** | `leadership_roles_count`, `team_captain`, `honors_thesis` | Campus impact |

Never collapse these into one hidden number. A single blended score conceals which
definition of success is doing the work, and they select different people.

**Excluded from the label:** alumni giving. Captured nowhere, scored nowhere. It is
a wealth proxy and including it would teach the graph to prefer applicants who
resemble affluent alumni.

---

## 2. The Priorities Conflict

Measured on the actual pool, 1,200 applicants, short lists of 285.

### 2.1 Pairwise correlation

| Pair | r | Reading |
| :--- | :--- | :--- |
| Net revenue vs Diversity | **−0.926** | Near-total opposition |
| Net revenue vs Academic strength | −0.267 | Merit discounting; merit vs academic rating is +0.662 |
| Yield vs Net revenue | −0.166 | Higher net price suppresses yield |
| Yield vs Academic strength | **+0.005** | No relationship in this pool. See 2.4 |

### 2.2 What maxing one priority produces

| Short list | Academic | Yield | Mean net price | First-gen | Low-income |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Max **Yield** | 3.05 | **0.761** | $24,444 | 21% | 27% |
| Max **Net revenue** | 2.86 | 0.188 | **$65,332** | 25% | **0%** |
| Max **Academic** | **4.13** | 0.271 | $22,714 | 21% | 15% |
| Max **Access** | 2.99 | 0.335 | $7,468 | 15% | **74%** |
| Whole pool | 3.07 | 0.270 | $30,478 | 20% | 18% |

### 2.3 Short list overlap

| | Yield | Net rev | Academic | Access |
| :--- | :--- | :--- | :--- | :--- |
| **Yield** | — | 14% | 24% | 33% |
| **Net rev** | 14% | — | 14% | **0%** |
| **Academic** | 24% | 14% | — | 20% |
| **Access** | 33% | **0%** | 20% | — |

**Applicants appearing in all four short lists: zero.**

This is the finding the dashboard exists to communicate. There is no universally
strong candidate. Four priorities produce four largely disjoint classes, and
maxing net revenue against access yields literally no one in common. The job is
choosing a blend, and the tool's purpose is making the cost of each blend visible.

### 2.4 Two honest caveats about the mock data

**Yield and academic strength show no correlation here.** Textbook expectation is
inverse, stronger applicants hold more offers and yield lower. The generator does
encode that, but merit awards and honours invitations to strong applicants offset
it almost exactly. Arguably realistic, since merit aid exists to do precisely
that, but do not present the absence of tension as a general finding.

**Academic strength does not track advantage in this pool, and real data will
differ.** Measured: first-generation applicants average 3.12 against 3.05 for
continuing-generation, and public school 3.10 against private-independent 2.97.
The generator drew academic strength independently of income band, so the
real-world correlation between socioeconomic advantage and measured credentials
is absent. Consequence: **the dashboard will under-display the Diversity versus
Academic Strength tension.** Either correct the generator or caption the panel.
Someone will otherwise demo this and conclude the tradeoff does not exist.

**Region count does not discriminate.** All four short lists span all seven
regions at n=285, so counting represented regions is useless as a diversity
metric. Diversity must be measured as distance from a target composition, per
SR-5.

---

## 3. Assumptions

- **A1.** Population one is `mock_admit_pool_fall2026.csv`, 1,200 rows, unchanged.
- **A2.** Population two is new. Roughly 600 alumni across six or more cohorts, about 200 flagged `exemplar_flag = Y`.
- **A3.** Mock data only, so no FERPA exposure at this stage.
- **A4.** Single-user local demo, no authentication. Acceptable only while data is synthetic. See NFR-4.
- **A5.** "Real time" means responsive to a control change, not streaming data.
- **A6.** Class target of 285 for short list sizing, consistent with the aid model.

---

## 4. Data Requirements

### DR-1 Current cycle applicants

Reuse `mock_admit_pool_fall2026.csv`. Relevant fields: `applicant_id`,
`intended_major`, `intended_school`, `academic_rating`, `hs_gpa_recalc`,
`region`, `hs_type`, `first_generation`, `family_income_band`, `computed_need`,
`net_price`, `predicted_yield_prob`, `athletic_recruit_sport`,
`coach_support_tier`, `special_talent_tag`, `demonstrated_interest_rating`.

### DR-2 Activity edge list — new, and the core gap

The pool carries exactly one `special_talent_tag` per applicant. An activity
graph needs many-to-many, so activities move to their own edge list spanning
**both** populations.

`activities_edges.csv`

| Column | Type | Notes |
| :--- | :--- | :--- |
| `person_id` | string | FK to `applicant_id` or `alum_id` |
| `person_type` | enum | `Applicant` / `Alum` |
| `activity_id` | string | FK to DR-3 |
| `context` | enum | `High School` / `Undergraduate` |
| `role` | enum | Member, Officer, Captain, President, Founder |
| `years_involved` | integer 1-4 | |
| `recognition_level` | enum | None, School, Regional, State, National |

Three to six activities per person, Zipf-distributed so common activities form
hubs and rare ones stay sparse. Uniform distribution produces a useless graph.

### DR-3 Controlled activity vocabulary

`activities_dim.csv`: `activity_id`, `activity_name`, `category`. One shared
vocabulary across both populations, otherwise nothing links.

- **Competitive intellectual** — Debate and Speech, Model UN, Math Olympiad, Science Olympiad, Robotics, Quiz Bowl
- **Athletics** — varsity by sport, club and intramural, see DR-5
- **Arts** — Instrumental Music, Vocal Music, Theatre, Dance, Studio Art, Film
- **Publications** — Newspaper, Yearbook, Literary Magazine, Radio
- **Governance** — Student Government, Class Officer, Honour Council
- **Service** — Community Volunteering, Tutoring and Peer Mentoring
- **Identity and culture** — Cultural and Affinity Organisations, International Student Association
- **Enterprise** — Entrepreneurship, Business and Investment, DECA
- **Research** — Independent Research, Published Work, Lab Assistantship
- **Obligation** — Part-time Employment, Family Caregiving

The last category matters. Paid work and caregiving displace the activities the
other nine reward. Omitting them makes the graph read under-involvement where the
real constraint was time, which directly undercuts P3.

### DR-4 Alumni outcome cohort — new

`alumni_outcomes.csv`, two blocks. The split is the whole point.

**Entry snapshot**, mirroring DR-1 field for field so comparison is like for like
at the point of admission: `entry_academic_rating`, `entry_hs_gpa_recalc`,
`entry_testing_policy`, `entry_sat_total`, `entry_region`, `entry_hs_type`,
`entry_first_generation`, `entry_family_income_band`, `entry_institutional_aid`,
`entry_intended_major`, `entry_athletic_recruit_sport`,
`entry_demonstrated_interest_rating`.

**Outcome block:** `cohort_year`, `final_major`, `final_school`,
`first_year_gpa`, `final_cum_gpa`, `gpa_delta`, `retained_year2`,
`graduated_4yr`, `graduated_6yr`, `academic_probation_ever`, `major_changes`,
`leadership_roles_count`, `varsity_seasons`, `team_captain`, `honors_thesis`,
`latin_honors`, `internships_count`, `study_abroad`, `postgrad_status_6mo`,
`success_attainment`, `success_growth`, `success_completion`,
`success_contribution`, `success_composite_tier`, `exemplar_flag`.

**DR-4a.** Applicants compare only against the entry block. Scoring an applicant
against an alum's final GPA is a category error; the applicant has no counterpart.

**DR-4b.** No alumni giving field. Not captured, per 1.4.

**DR-4c.** The four `success_*` components are stored separately, per 1.4.

### DR-5 Athletics

Varsity participation is both activity and attribute. Model it once, in DR-2,
sport as `activity_id`, `role` distinguishing captain from squad. `varsity_seasons`
in DR-4 is a convenience rollup. Two copies of the sport will disagree.

### DR-6 Campus clubs — new

Campus clubs are **not** the same entity as DR-3 activities, and the distinction
carries the feature. DR-3 is a vocabulary of things people did, mostly before
college. DR-6 is the roster of organisations that exist *here*, each with a
membership count, a graduating cohort, and a viability floor.

That difference creates a link nothing else in the model provides: an applicant's
interest in a campus club connects forward to alumni who were in that club and
thrived, and simultaneously connects to whether the club will still have members
next year.

`campus_clubs_dim.csv`

| Column | Type | Notes |
| :--- | :--- | :--- |
| `club_id` | string | |
| `club_name` | string | |
| `category` | enum | Reuses the DR-3 categories |
| `linked_activity_id` | string, nullable | FK to DR-3. **The bridge.** High school Debate maps to the campus Debate Society |
| `current_members` | integer | |
| `graduating_members` | integer | Members leaving with this cohort |
| `min_viable_members` | integer | Below this the club cannot function |
| `charter_status` | enum | Active, Probationary, Dormant |
| `advisor_backed` | Y/N | Faculty advisor secured |

`club_edges.csv` — one file spanning both populations, matching the DR-2 pattern

| Column | Type | Applies to |
| :--- | :--- | :--- |
| `person_id` | string | Both. FK to `applicant_id` or `alum_id` |
| `person_type` | enum | `Applicant` / `Alum` |
| `club_id` | string | Both. FK to `campus_clubs_dim` |
| `relationship` | enum | `INTERESTED_IN` for applicants, `MEMBER_OF` for alumni |
| `interest_signal` | enum, nullable | Applicants only: Portal Click, Info Session Topic, Campus Visit Meeting, Essay Mention, Application Checkbox, Counsellor Note |
| `signal_strength` | enum, nullable | Applicants only: Passive, Explicit, Sustained |
| `club_role` | enum, nullable | Alumni only: Member, Officer, President, Founder |
| `years_active` | integer, nullable | Alumni only, 1-4 |

**DR-6a.** Columns are nullable by `relationship`. The two relationships carry
genuinely different attributes and forcing them into shared columns would require
meaningless values.

**DR-6b.** `signal_strength` is a word scale in the data, not a number, consistent
with IR-1. The interface must never have to render a club interest score.

**DR-6c.** Generation must produce deliberate scarcity. Some clubs need to be at
genuine succession risk, with `current_members - graduating_members` falling below
`min_viable_members` and few or no interested applicants. If every club is healthy
the feature has nothing to show. Target roughly a fifth of clubs at risk.

**DR-6d.** Applicant club interest must correlate with, but not be determined by,
their DR-2 high school activities through `linked_activity_id`. Perfect
correlation makes the club layer redundant; zero correlation makes it noise.

### NR-1 Node types

| Type | Source | Visual treatment |
| :--- | :--- | :--- |
| Applicant | DR-1 | Distinct shape, one colour family |
| Alum | DR-4 | Distinct shape, second family, exemplars emphasised |
| Activity | DR-3 | Neutral, size by degree |
| Major | `intended_major` / `final_major` | Neutral, distinct shape |
| Athletic team | DR-3 athletics | Neutral, distinct shape |
| Campus club | DR-6 | Neutral, distinct shape, sized by roster health, outlined when at succession risk |

Applicant and Alum must be distinguishable by shape as well as colour. Colour
alone fails colour-blind users, see NFR-3.

### NR-2 Edge types

| Edge | Between | Meaning |
| :--- | :--- | :--- |
| `PARTICIPATES_IN` | Person to Activity | DR-2, weighted by role and recognition |
| `STUDIES` | Person to Major | Intended for applicants, final for alumni |
| `PLAYS_FOR` | Person to Athletic team | Varsity only |
| `MEMBER_OF` | Alum to Campus club | Observed history, DR-6, weighted by role |
| `INTERESTED_IN` | Applicant to Campus club | Stated or inferred intent, DR-6, weighted by signal strength |
| `SIMILAR_TO` | Applicant to Alum | Computed, section 6 |

**NR-2a.** Three visual tiers, because the epistemic status differs and conflating
them is how a graph lies:

| Tier | Edges | Treatment |
| :--- | :--- | :--- |
| Observed past | `PARTICIPATES_IN`, `STUDIES`, `PLAYS_FOR`, `MEMBER_OF` | Solid |
| Observed intent | `INTERESTED_IN` | Dashed |
| Inferred | `SIMILAR_TO` | Distinct styling, clearly not an observation |

`INTERESTED_IN` is a real signal about an unrealised future, not a fact about the
past. It must never render identically to `MEMBER_OF`.

### NR-3 Rendering scale

1,200 applicants against 600 alumni is 720,000 candidate pairs. Rendering all of
them is a hairball with no information content.

- **NR-3a.** At most `k` `SIMILAR_TO` edges per applicant, default 3, set by a word-labelled control.
- **NR-3b.** Suppress edges below a relevance floor even when under `k`. A weak top match is not a match.
- **NR-3c.** Default view is a focused subgraph. Full-population view must be explicitly requested, with a warning.
- **NR-3d.** Focus modes: one applicant and neighbourhood, one activity and everyone attached, one major cohort, one campus club and its alumni and interested applicants, all clubs at succession risk, current short list.

### NR-4 Cohort clusters, legible from a distance

The graph currently arranges applicants and alumni in opposing arcs with
attribute nodes in central bands. That arrangement is deliberately **not**
grouped by profile, and the consequence is that a wide view reads as an
undifferentiated mass of nodes and edges. It shows connectivity and hides
structure.

Replace it. At the widest view the dominant visual feature must be a small number
of **named cohort clusters**, each large enough to read and labelled in place.

- **NR-4a.** Both populations are assigned to cohorts by the **same rules**, applicants on their current record and alumni on their `entry_*` block. This is the payoff: an athlete cluster of applicants sitting beside the athlete cluster of alumni who already graduated makes "did people like this thrive here" a visual question rather than a query.
- **NR-4b.** Cohorts are **rule-based and named**, never discovered. See NR-4f.
- **NR-4c.** Applicants and alumni remain distinguishable **within** a cohort region, per NR-1. Clustering must not cost the population distinction.
- **NR-4d.** A person belongs to many cohorts at once, but a point has one position. Each is assigned a **primary cohort by disclosed precedence**, with full membership shown on selection. Undisclosed precedence makes cluster membership look arbitrary.
- **NR-4e.** A cohort falling below a minimum size merges into the residual cohort. A three-person cluster reads as noise, not structure.
- **NR-4f.** The cohort vocabulary, each definition, and the precedence order:

| Cohort | Definition | Both populations via |
| :--- | :--- | :--- |
| Recruited athletes | A sport is present | `athletic_recruit_sport` / `varsity_seasons > 0` |
| Performing and visual arts | Fine Arts school, or an arts activity carrying State or National recognition | school, DR-2 edges |
| Pre-law and civic | Political Science, History or English **and** one of Debate and Speech, Model UN, Student Government, Class Officer | major + DR-2 edges |
| Research-track sciences | Biology, Chemistry, Neuroscience, Environmental Science or Mathematics **and** one of Independent Research, Published Work, Lab Assistantship | major + DR-2 edges |
| Engineering and applied science | School of Engineering | school |
| Health sciences | School of Nursing and Health | school |
| Business and enterprise | School of Business | school |
| Top academic | Reader rating 5, or recalculated GPA at or above 3.90 | rating, GPA |
| International | International citizenship or region | citizenship / `entry_region` |
| Access and first-generation | First-generation **and** an income band below $60K | first-gen, income |
| Highly engaged | Demonstrated interest rating 4 or 5 | interest rating |
| General cohort | Everything else, plus any cohort merged under NR-4e | residual |

Pre-law and Research-track are **conjunctions** of a major and an activity, which
is exactly the kind of derived definition XR-12 requires be stated on screen.
Neither half alone qualifies.

**NR-4g Precedence is computed by ascending predicate breadth, rarest first.**

Not hand-ordered. A hand-picked order was tried and the data disproved it: 64
people matched Access and first-generation, but only 10 reached it as their
primary, because broad school-track predicates upstream claimed the rest. The
cohort then fell below the NR-4e floor and vanished, breaking FR-7's requirement
that it ship visible.

Rarity ordering fixes that without anyone deciding which group matters more. It is
a legibility rule rather than a policy judgement: the narrowest predicates get
first claim, so the most distinctive groups stay large enough to read. It also
maintains itself if the data changes, and it is deterministic for a given dataset,
which is all NFR-6 requires.

The computed order must be disclosed under XR-12, since it is no longer a
constant. On the current data it runs Access and first-generation, Research-track
sciences, Pre-law and civic, Recruited athletes, International, Top academic,
Health sciences, Engineering, Performing and visual arts, Business, Highly
engaged, General.

One consequence to state plainly: a recruited athlete who is also first-generation
and low-income lands in **Access**, not Athletes, because Access is the rarer
predicate. Full membership is retained and shown on selection per NR-4d, so the
athlete is still findable; only their position is single-valued.

### NR-5 Progressive disclosure of edges

At the widest view, edges must not be the dominant visual element. As the view
narrows they must become individually legible.

- **NR-5a.** At the widest view the edge layer reads as texture, not as lines. Cohort clusters and their labels dominate.
- **NR-5b.** As the view narrows, individual edges resolve and their tier becomes distinguishable per NR-2a.
- **NR-5c.** `SIMILAR_TO` edges are the last to appear, and only for a focused selection. They are inferred rather than observed, and a wide view has no business asserting inference.
- **NR-5d.** Shared-path highlighting under XR-6 **overrides** disclosure at every level. If a user asks why two people are connected, the answer is drawn regardless of how far out the view sits.

**A feasibility constraint, stated rather than discovered later.** Streamlit's
chart component surfaces only point, box and lasso selection events. It does
**not** report pan or zoom, so the server cannot know the current zoom level.
Continuous zoom-driven level-of-detail is therefore **not achievable** through
the supported event model. Three mechanisms, with honest labels:

| Mechanism | Achievable | Note |
| :--- | :--- | :--- |
| Density-driven opacity: edges drawn thin and faint, so a wide view reads as a wash and a narrow view resolves individual lines | **Yes**, no events needed | The primary mechanism. Perceived density falls as the same edges spread over more screen area |
| A discrete detail-level control, word-scaled per IR-1 | **Yes** | The explicit override, see IR-8 |
| True continuous zoom-driven level-of-detail | **No**, not in Streamlit's event model | Needs a custom JavaScript component. Out of scope unless that scope is agreed |

**NR-5e.** The two achievable mechanisms are required. The third must not be
promised in the interface, and no control may imply the graph is responding to
zoom when it is responding to a setting.

---

## 6. Scoring Requirements

### SR-1 Decomposability is mandatory

Both the similarity score and the priority composite must be additive weighted
means over named components:

```
score = sum_d ( w_d * component_d ) / sum_d ( w_d )
```

Component `d` contributes `w_d * component_d / sum_d(w_d)`, and contributions sum
exactly to the total. This is what makes section 8 possible.

**SR-1a.** Graph neural networks, learned embeddings, and matrix factorisation are
**out of scope**. Not because they would score worse, but because their output
cannot be decomposed into reasons a committee can interrogate. Explainability was
stated as the priority and it constrains the method.

**SR-1b.** Normalise by `sum_d(w_d)` so scores stay comparable as weights change.
Without it, enabling a component inflates every score and the graph appears to
improve for no reason.

### SR-2 Similarity dimensions, applicant to alum

| Dimension | Compares | Method |
| :--- | :--- | :--- |
| Academic preparation | Entry rating, recalculated GPA, rigour | Banded distance |
| Field of study | Intended major against final major | Exact, then same-school partial credit |
| Activity portfolio | Shared activity nodes | Weighted Jaccard, role and recognition as weights |
| Athletics | Varsity participation, sport, captaincy | Tiered match |
| Leadership depth | Officer and captain roles | Banded distance |
| School and geography | HS type, region, feeder history | Categorical |
| Socioeconomic context | First-generation, income band, aid | Categorical, see FR-2 |
| Engagement | Demonstrated interest rating | Banded distance |
| Campus club affinity | Applicant `INTERESTED_IN` set against alum `MEMBER_OF` set | Rarity-weighted Jaccard, bridged by `linked_activity_id`, see SR-7 |

### SR-3 Excluded inputs

`ethnicity_ipeds` and `gender` must never enter any component, appear as a
dimension, or be inferrable from a displayed contribution. Retained for reporting
only.

### SR-4 Activity weighting by rarity

Shared membership in a hub activity is weak evidence; shared membership in a rare
one is strong. Weight overlap by inverse activity frequency, or every applicant
matches every alum through Community Volunteering.

### SR-5 Diversity is set-dependent, not individual

P1, P2, and P4 are individual scores, so a short list is a sort. **P3 is not.** No
individual is diverse; an applicant contributes to composition relative to who is
already selected. Two consequences:

- **SR-5a.** P3 is scored as marginal reduction in distance between projected class composition and stated composition targets, across the DR-1 dimensions listed in 1.2 P3.
- **SR-5b.** When P3 carries any weight, the short list must be built **greedily**, recomputing P3 after each selection. Sorting on a precomputed P3 column is wrong, since each pick changes the value of every remaining candidate.
- **SR-5c.** Composition targets are user-editable and must be stated in the interface. A diversity score against undisclosed targets is not explainable.

### SR-6 Expected, not nominal

P2 uses expected NTR, `net_price` × `predicted_yield_prob`. P1 and P2 therefore
share a factor and are not independent; state this at the controls.

### SR-7 Club affinity, and matching on absence

Applicant club interest is matched against alum club membership, rarity-weighted
per SR-4 so shared interest in a large club counts for less than a small one.

Two rules that are easy to get wrong:

- **SR-7a.** Interest in a club whose `linked_activity_id` the applicant already pursued in high school scores higher than interest alone. Stated intent backed by four years of Debate is stronger evidence than a portal click.
- **SR-7b.** Two people sharing *no* clubs must not score as similar. Jaccard over two empty sets is undefined, and treating absence as agreement silently inflates similarity across the entire low-engagement population. Where both sets are empty the dimension is **dropped from that pair's normaliser**, not scored zero and not scored one. The same rule applies to the Athletics dimension in SR-2.

### SR-8 Club succession, and the case for the non-ideal candidate

This is what the club layer is for. Projected roster for each club:

```
projected = current_members - graduating_members
          + sum over interested applicants ( predicted_yield_prob )
```

- **SR-8a.** A club is **at succession risk** when `projected < min_viable_members`. Flag it on the node.
- **SR-8b.** Identify **sole-pipeline applicants**: an applicant is the only, or one of very few, interested parties for an at-risk club. These are frequently applicants who rank nowhere on P1, P2, or P4.
- **SR-8c.** Club coverage is a composition dimension inside **P3**, not a fifth priority. It scores as marginal reduction in the count of at-risk clubs left uncovered, and is therefore set-dependent and subject to SR-5b greedy selection.
- **SR-8d.** Roster projection is probability-weighted, so it inherits every caveat attached to `predicted_yield_prob`. Display it as a range or a qualitative band, never as a confident headcount.

**Why this matters.** The four priorities in section 1.2 are all optimised by
admitting people who are already strong on a measurable axis. None of them can
express "the radio station loses six seniors and this is the only applicant who
wants to run it." Club succession gives a documented, auditable reason to admit
outside the top of the composite, which is the one thing a pure four-priority
ranking cannot produce.

**A note on scope.** Campus vitality is arguably a fifth institutional priority
and could carry its own control. It is specified inside P3 because the four
priorities were stated as the model, and quietly adding a fifth would change the
agreed objective function. Promoting it is a small change and an explicit
decision, not a default.

---

## 7. Interaction Requirements

### IR-1 No numbers on controls

Every importance control is an ordinal word scale. No integers, percentages, or
decimals in the widget, its label, or its tooltip.

```
Not considered  ·  Minor  ·  Moderate  ·  Major  ·  Defining
```

Implemented with `st.select_slider` over strings. `st.slider` is prohibited for
weights; it renders numeric values.

**IR-1a.** The numeric mapping must be monotone increasing and documented in the
repository, never surfaced in the interface.

**IR-1b.** `Not considered` means a true zero weight, dropping the component from
the normaliser. If staff say a factor is not considered, it must not leak in.

### IR-2 The four priority controls

One word-scaled control per priority: **Yield**, **Net Tuition Revenue**,
**Diversity**, **Academic Strength**. Setting any one to `Defining` while the rest
sit at `Not considered` must reproduce the corresponding column of the 2.2 table.
That equivalence is the demo.

### IR-3 Short list selection

- **IR-3a.** The short list is the top of the composite ranking, sized by a word-scaled control: `Very selective` · `Selective` · `Broad`, against the A6 class target.
- **IR-3b.** Short list members are highlighted as a selected node group in the graph, per the request to see the group light up.
- **IR-3c.** Selecting a short list member reveals its **evidence subgraph**: the alumni, activities, majors, and teams that justify the pick. Both readings of "the group of nodes for each student" are in scope, the selected cohort and the supporting evidence, since they answer different questions.
- **IR-3d.** Short list membership must be exportable, per XR-9.

### IR-4 Secondary controls

Per-dimension toggles and importance for SR-2. Match strictness, driving NR-3b.
Matches per applicant, driving NR-3a: `Closest only` · `A few` · `Several` ·
`Many`. Success-label component toggles per 1.4. Composition targets per SR-5c.
Population filters: intended school, region, disposition, exemplars only.

### IR-5 Presets

Named configurations setting every control at once, each with a written
description of the philosophy it encodes: `Yield-first`, `Revenue-first`,
`Access and mission`, `Academic-first`, `Balanced`. Presets are the fastest way to
demonstrate that the graph is a lens, not a verdict.

### IR-6 Reset, and pin

Return to last preset, return to defaults. Pin up to four applicants and hold
them visible through reconfiguration, so staff watch specific people move rather
than an anonymous cloud shifting.

### IR-7 Club controls

- **IR-7a.** Show or hide campus club nodes. Clubs add a dense layer and must be dismissible.
- **IR-7b.** Filter to clubs at succession risk only, per SR-8a.
- **IR-7c.** A word-scaled control for how much club coverage weighs inside P3: `Not considered` · `Minor` · `Moderate` · `Major` · `Defining`. Subordinate to the P3 control, and inert when P3 is `Not considered`.
- **IR-7d.** Selecting a club node reveals its roster projection, its graduating cohort, the alumni who were members with their outcomes, and every interested applicant.
- **IR-7e.** A control to surface sole-pipeline applicants, per SR-8b, independent of their composite rank. This is the one path in the interface that deliberately promotes applicants the composite ranks low, and it must be labelled as such.

### IR-8 Detail level and cluster controls

- **IR-8a.** A word-scaled detail control, per IR-1, driving NR-5: `Clusters only` · `Outlines` · `Connections` · `Everything`. No numbers, and no label implying it reflects zoom.
- **IR-8b.** Cohort labels can be shown or hidden. At the widest view they are on by default, since they are the point of NR-4.
- **IR-8c.** A cohort can be isolated, dimming every other cluster, as an eighth focus mode under NR-3d.
- **IR-8d.** Cohort assignment must be switchable off, returning the population-arc arrangement. Clustering imposes one reading of the pool, and a user must be able to remove it.
- **IR-8e.** Selecting a cluster reports its size, its definition in words, and its composition against the P3 targets, so a cluster is inspectable rather than merely decorative.

---

## 8. Explainability Requirements

The stated priority. Each is a hard requirement.

**XR-1 Contributions sum to the whole.** Selecting a `SIMILAR_TO` edge or a short
list member shows every component's contribution as proportional bars that
visibly total the score.

**XR-2 Written rationale.** Every edge and every short list pick carries a
generated sentence naming top contributors in plain language. For example:
*"Shortlisted mainly for expected net revenue, with a shared Debate and Speech
background linking to three alumni who graduated with honours."* Generated from
the SR-1 decomposition, never hand-authored.

**XR-3 No raw coefficients by default.** Internal weights sit behind an explicit
disclosure control, consistent with IR-1.

**XR-4 Show what changed.** After any control change, report which applicants
entered the short list, which left, and which priority caused it. A graph that
silently rearranges teaches nothing. This requirement is what makes word-scaled
controls legible: with no numbers, the only way to understand a control is to
observe its effect.

**XR-5 Show the cost of the choice.** Whenever one priority is set to `Defining`,
display the realised values of the other three against the whole-pool baseline,
in the form of the 2.2 table. Maxing net revenue must visibly report that
low-income representation went to zero. A tool that lets staff max one priority
without showing what it cost is not advisory, it is persuasive.

**XR-6 Shared-path highlighting.** Selecting an applicant-alum pair highlights the
observed nodes they actually share, so evidence is visible in the graph rather
than asserted in a tooltip. This is the payoff for using a graph at all.

**XR-7 Nearest exemplars.** Per applicant, the top matching alumni with each one's
headline reason and outcome summary.

**XR-8 Counterfactual.** Per applicant, what would most change their standing, for
example *"Enters the short list if Diversity moves above Minor."* Framed as
explanation of the model, never as advice to applicants.

**XR-9 Configuration always legible.** The active configuration rendered as a
sentence, always on screen: *"Prioritising net tuition revenue above all, with
academic strength moderate, ignoring yield and diversity."*

**XR-10 Exportable rationale.** Export any applicant's comparison set, scores,
decomposition, short list status, and the active configuration, for the committee
record.

**XR-11 Reproducibility.** Every displayed score recomputable from the displayed
configuration alone. No hidden state, no cross-session persistence.

**XR-12 Cluster definitions are stated, not implied.** A cluster label is an
assertion about people, and a reader will treat spatial grouping as fact. Every
cohort's rule from NR-4f must be readable in the interface, alongside the
precedence order that resolved multiple membership. Cohorts 3 and 4 are
conjunctions of a major and an activity, so "Pre-law and civic" must not be
allowed to read as a self-evident category.

**XR-13 Disclosure level is never silent.** The active detail level is stated in
the configuration sentence under XR-9. A user looking at a view with no visible
edges must be able to tell whether the graph has no edges, or is not currently
drawing them.

---

## 9. Fairness and Governance

Similarity-to-past-success is a bias amplifier by construction. If the exemplar
cohort over-represents a school type, region, or income band, the graph promotes
applicants resembling them and calls it merit. That mechanism is the intended
behaviour, which is exactly why it needs instrumentation.

**FR-1 Cohort composition panel.** Always available. Exemplar cohort composition
across region, HS type, first-generation status, and income band, beside the
applicant pool's. Visible skew is the warning the tool owes its user.

**FR-2 Socioeconomic dimension is reviewable.** First-generation and income
context can serve access goals or entrench advantage depending on sign and
company. Ships **off** by default in SR-2, with the tradeoff stated at the
control. Note it is distinct from P3, which is a class-composition objective
rather than a similarity dimension.

**FR-3 Outcome drift warning.** Warn when the selected success definition
correlates strongly with entry advantage, for example exemplars clustering in top
income bands. An attainment-only label frequently does this.

**FR-4 Advisory only.** Confirmed as a requirement, not a default. No ranked admit
or deny output, no automated cut, no field named or styled as a decision. The
short list is explicitly labelled as a discussion set. Every export carries the
configuration that produced it, so no number travels without its reasoning.

**FR-5 Suppression audit.** The interface must confirm that ethnicity and gender
are excluded, per SR-3, so exclusion is verifiable rather than assumed.

**FR-6 Mock data caveat surfaced in the interface.** The limitations in 2.4 must
appear in the app, not only in this document. The Diversity versus Academic
Strength panel understates real tension and a demo audience will not know that.

**FR-7 No cohort may be defined on a protected attribute, or on a proxy for
one.** This is the most consequential constraint the NR-4 change introduces, and
it is new: spatial clustering turns a scoring rule into a picture. A cohort
defined on ethnicity or sex would render the applicant pool as a segregation map,
visible at a glance and impossible to unsee, and the fact that SR-3 keeps those
fields out of *scoring* would not save it. Specifically:

- Ethnicity and sex are absent from the loaded data entirely, per SR-3, so no cohort can reference them.
- `proxy_sensitive` club membership must not define or influence a cohort, per §0.1. Interest in an affinity or religious organisation is a proxy for race or religion, and a cluster built on it is the same map by another name.
- Cohort 10, Access and first-generation, is defined on first-generation status and income band. Those are legitimate mission categories and P3 already scores them, but a *visible cluster* is a stronger act than a scoring weight. It ships **visible and removable** under IR-8d, and FR-1's composition panel must cover it.

**FR-8 Clusters must not become a ranking.** Cohorts are a way of reading the
pool, not a quality order. Their arrangement on screen must not imply rank: no
left-to-right or top-to-bottom ordering by desirability, no colour ramp from good
to poor. Position within the canvas carries no meaning beyond grouping.

---

## 10. Non-Functional Requirements

**NFR-1 Responsiveness.** Control change to redrawn graph under 300 ms at stated
scale. Precompute per-dimension match matrices once, vectorised, and cache; only
the weighted combination recomputes on interaction. Reweighting must never
recompute pairwise dimension scores. Note SR-5b greedy selection is the one path
that cannot be a pure reweighting, so budget for it.

**NFR-2 Graph library.** Must support click or select events on nodes and edges,
stable layout across redraws, and NR-3 scale. Layout instability defeats XR-4,
since everything appears to move.

Resolved for this build: Plotly on Streamlit 1.61.1, with one trace per edge tier
and numpy coordinate arrays. Verified at NR-3 scale.

Two limits found by measurement rather than assumption:

- **Pan and zoom are not reported to the server.** Only point, box and lasso selection are. This is what makes NR-5's third mechanism unachievable, and it is a property of the component, not of the chosen layout.
- **Selection events are wired but unproven.** The parameters exist and the plumbing is written; that events actually fire on a node click needs a browser and has not been confirmed.

**NFR-6 Clustered layout stays frozen.** NR-4 changes how positions are computed,
not how often. Cohort assignment depends only on the loaded data, never on a
control value, so positions are computed once and cached exactly as before. If
cohort assignment ever became weight-dependent, every node would move on every
interaction and XR-4 would be defeated, so it must not.

**NFR-3 Accessibility.** Shape plus colour for node types, never colour alone.
Keyboard-reachable controls. Every visual claim must also exist in words, which
XR-2 and XR-9 already require.

**NFR-4 Security.** No authentication in scope while data is synthetic. Real
applicant or student records would make this a FERPA system requiring access
control, audit logging, and transport security before any deployment beyond
localhost. Do not bind to a public interface with real data.

**NFR-5 Determinism.** Seeded generation. Identical configuration yields identical
output, per XR-11.

---

## 11. Out of Scope

Automated admit or deny recommendations. Model training or validation against real
outcomes. Real student data. Multi-user state, authentication, deployment.
Predicting individual student GPA as a published figure. Aid optimisation, covered
by `aid_allocation_model.py` and `aid_leveraging_algorithm.md`.

---

## 12. Deliverables

| Path | Purpose |
| :--- | :--- |
| `requirements.md` | This document |
| `generate_mock_graph_data.py` | Builds DR-2, DR-3, DR-4, seeded |
| `alumni_outcomes.csv` | DR-4 |
| `activities_edges.csv` | DR-2 |
| `activities_dim.csv` | DR-3 |
| `campus_clubs_dim.csv` | DR-6 |
| `club_edges.csv` | DR-6 |
| `clubs.py` | SR-7 affinity, SR-8 succession projection |
| `*.schema.md` | Data dictionary per file, matching project convention |
| `priorities.py` | P1-P4 scoring and greedy short list, SR-5 |
| `similarity.py` | SR-1, SR-2, SR-4 |
| `app.py` | Dashboard |
| `requirements.txt` | Pinned dependencies |

`priorities.py` and `similarity.py` must not import Streamlit, so scoring is
testable without launching the interface.

---

## 13. Acceptance Criteria

1. Both populations render in one graph, distinguishable by shape and colour.
2. Applicants and alumni connect through shared activity, major, and athletic team nodes, and those shared nodes are highlightable, per NR-2 and XR-6.
3. Debate and Speech, university majors, varsity athletics, and campus clubs all appear as first-class nodes with real edges.
4. `SIMILAR_TO` edges carry scores and are visually distinct from observed edges.
5. Four word-scaled priority controls exist for Yield, Net Tuition Revenue, Diversity, and Academic Strength.
6. Setting exactly one priority to `Defining` reproduces the corresponding row of the 2.2 table, within greedy-selection tolerance for Diversity.
7. The short list is highlighted as a selected node group, and each member exposes its evidence subgraph, per IR-3b and IR-3c.
8. No control anywhere displays a number for importance, per IR-1.
9. `Not considered` provably removes a component from the score, per IR-1b.
10. Every score decomposes into contributions summing to the total, per XR-1.
11. Every edge and short list pick carries a generated plain-language rationale, per XR-2.
12. Any control change updates the graph in under 300 ms and reports what changed, per XR-4 and NFR-1.
13. Maxing one priority displays the realised cost to the other three against baseline, per XR-5.
14. Diversity short lists are built greedily, and sorting on a static diversity column is demonstrably absent, per SR-5b.
15. Composition targets are visible and editable, per SR-5c.
16. Active configuration is legible as a sentence at all times, per XR-9.
17. Exemplar cohort composition is inspectable beside the applicant pool, per FR-1.
18. Ethnicity and gender are absent from all scoring paths, verifiable in the interface, per SR-3 and FR-5.
19. The 2.4 mock data caveats appear in the interface, per FR-6.
20. Identical configuration reproduces identical scores, per NFR-5 and XR-11.
21. Nothing in the interface is styled or named as a decision, per FR-4.
22. Campus club nodes render with roster health visible and succession risk flagged, per NR-1 and SR-8a.
23. `INTERESTED_IN` and `MEMBER_OF` edges are visually distinct from each other and from `SIMILAR_TO`, per NR-2a.
24. Selecting a club exposes its roster projection, graduating cohort, member alumni with outcomes, and interested applicants, per IR-7d.
25. At least one club is at genuine succession risk in the generated data, per DR-6c, and its sole-pipeline applicants are identifiable, per SR-8b.
26. Pairs sharing no clubs have the club dimension dropped from their normaliser rather than scored, per SR-7b.
27. Club coverage influences the short list only through P3 and goes inert when P3 is `Not considered`, per SR-8c and IR-7c.
28. At the widest view, named cohort clusters are the dominant visual feature and each carries a readable in-place label, per NR-4 and NR-4f.
29. Applicants and alumni are assigned to cohorts by the same rules, so each cohort contains both populations and they remain distinguishable within it, per NR-4a and NR-4c.
30. Every person has exactly one primary cohort, resolved by the published precedence, with full membership visible on selection, per NR-4d.
31. No cohort falls below the minimum size; undersized cohorts merge into the residual, per NR-4e.
32. At the widest view the edge layer reads as texture rather than lines, and individual edges resolve as the view narrows, per NR-5a and NR-5b.
33. `SIMILAR_TO` edges appear only for a focused selection, never in the widest view, per NR-5c.
34. Shared-path highlighting overrides disclosure at every detail level, per NR-5d.
35. The detail control is word-scaled, and no label implies the graph is responding to zoom, per IR-8a and NR-5e.
36. Cohort clustering can be switched off, restoring the population-arc arrangement, per IR-8d.
37. Every cohort definition and the precedence order are readable in the interface, per XR-12.
38. The active detail level appears in the configuration sentence, so an edgeless view is never ambiguous, per XR-13.
39. No cohort references ethnicity, sex, or `proxy_sensitive` club membership, verifiable in the interface, per FR-7.
40. Cluster arrangement encodes no ranking: no ordering by desirability and no good-to-poor colour ramp, per FR-8.
41. Cohort assignment depends only on loaded data, so the layout is still computed once and cached, per NFR-6.
