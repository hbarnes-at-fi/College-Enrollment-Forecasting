# Schema: `alumni_outcomes.csv`

600 rows x 39 columns. The DR-4 outcome cohort. Two blocks, and the split is the point.

**Entry block** (`entry_*`) mirrors the admit pool field for field. Applicants are compared only against this block; scoring an applicant against an alum's final GPA is a category error, since the applicant has no counterpart to it (DR-4a).

**Outcome block** carries the four success components separately (DR-4c). They are never collapsed into one hidden number, because a single blended score conceals which definition of success is doing the work, and the four definitions select different people.

**No giving field.** Alumni giving is a wealth proxy and is not captured anywhere (DR-4b).

Generated from the CSV by `data/write_schemas.py`; the examples are real rows, not illustrations.

| Column | Type | Example A | Example B |
| :--- | :--- | :--- | :--- |
| `alum_id` | string | L2019-10000 | L2020-10001 |
| `cohort_year` | integer | 2019 | 2020 |
| `entry_academic_rating` | integer | 2 | 4 |
| `entry_hs_gpa_recalc` | decimal | 3.25 | 3.85 |
| `entry_testing_policy` | enum | Submitted | Submitted |
| `entry_sat_total` | integer, nullable | 1200 | 1320 |
| `entry_region` | enum | West | International |
| `entry_hs_type` | enum | Public | Public |
| `entry_first_generation` | Y/N | N | N |
| `entry_family_income_band` | enum | $100K - $149K | $60K - $99K |
| `entry_institutional_aid` | integer | 13700 | 41700 |
| `entry_intended_school` | enum | College of Arts & Sciences | School of Business |
| `entry_intended_major` | string | History | Accounting |
| `entry_athletic_recruit_sport` | string, nullable |  |  |
| `entry_demonstrated_interest_rating` | integer | 2 | 2 |
| `final_school` | enum | College of Arts & Sciences | School of Business |
| `final_major` | string | Chemistry | Accounting |
| `first_year_gpa` | decimal | 2.85 | 3.23 |
| `final_cum_gpa` | decimal | 3.18 | 3.41 |
| `gpa_delta` | decimal | 0.33 | 0.18 |
| `retained_year2` | Y/N | Y | Y |
| `graduated_4yr` | Y/N | Y | Y |
| `graduated_6yr` | Y/N | Y | Y |
| `academic_probation_ever` | Y/N | Y | Y |
| `major_changes` | integer | 0 | 3 |
| `leadership_roles_count` | integer | 0 | 3 |
| `varsity_seasons` | integer | 0 | 0 |
| `team_captain` | Y/N | N | N |
| `honors_thesis` | Y/N | N | N |
| `latin_honors` | Y/N | N | N |
| `internships_count` | integer | 0 | 1 |
| `study_abroad` | Y/N | N | Y |
| `postgrad_status_6mo` | enum | Graduate or professional school | Graduate or professional school |
| `success_attainment` | decimal | 0.672 | 0.764 |
| `success_growth` | decimal | 0.7624 | 0.6139 |
| `success_completion` | decimal | 0.85 | 0.85 |
| `success_contribution` | decimal | 0.0 | 0.5455 |
| `exemplar_flag` | Y/N | N | Y |
| `success_composite_tier` | enum | Tier 3 - Solid | Tier 2 - Strong |

## Enum values

- `entry_testing_policy`: Submitted, Test Optional
- `entry_region`: International, Mid-Atlantic, Midwest, New England, South, Southwest, West
- `entry_hs_type`: Charter, Home School, International School, Parochial, Private - Independent, Public
- `entry_first_generation`: N, Y
- `entry_family_income_band`: $100K - $149K, $150K - $249K, $250K and above, $30K - $59K, $60K - $99K, Not Reported, Under $30K
- `entry_intended_school`: College of Arts & Sciences, College of Fine Arts, School of Business, School of Engineering, School of Nursing & Health
- `final_school`: College of Arts & Sciences, College of Fine Arts, School of Business, School of Engineering, School of Nursing & Health
- `retained_year2`: N, Y
- `graduated_4yr`: N, Y
- `graduated_6yr`: N, Y
- `academic_probation_ever`: N, Y
- `team_captain`: N, Y
- `honors_thesis`: N, Y
- `latin_honors`: N, Y
- `study_abroad`: N, Y
- `postgrad_status_6mo`: Employed full-time, Fellowship, Graduate or professional school, Seeking, Unknown
- `exemplar_flag`: N, Y
- `success_composite_tier`: Tier 1 - Thrived, Tier 2 - Strong, Tier 3 - Solid, Tier 4 - Struggled
