# Schema: `activities_edges.csv`

8,851 rows x 7 columns. The DR-2 many-to-many activity edge list, spanning both populations.

The admit pool carries exactly one `special_talent_tag` per applicant. An activity graph needs many-to-many, which is why activities live here rather than in a column.

`role` and `recognition_level` weight the edge in the SR-4 rarity-weighted Jaccard. Shared membership in a hub activity is weak evidence; shared membership in a rare one is strong.

Generated from the CSV by `data/write_schemas.py`; the examples are real rows, not illustrations.

| Column | Type | Example A | Example B |
| :--- | :--- | :--- | :--- |
| `person_id` | string | A26-101831 | A26-101831 |
| `person_type` | enum | Applicant | Applicant |
| `activity_id` | string | ACT-31 | ACT-12 |
| `context` | enum | High School | High School |
| `role` | enum | Member | Member |
| `years_involved` | integer | 4 | 4 |
| `recognition_level` | enum | School | State |

## Enum values

- `person_type`: Alum, Applicant
- `context`: High School, Undergraduate
- `role`: Captain, Founder, Member, Officer, President
- `recognition_level`: National, None, Regional, School, State
