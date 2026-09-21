# Schema: `club_edges.csv`

3,419 rows x 8 columns. The DR-6 club edge list. One file, two relationships, discriminated by `relationship`.

`INTERESTED_IN` (applicants) and `MEMBER_OF` (alumni) carry genuinely different attributes, so columns are nullable by relationship (DR-6a). The loader rejects any row that violates this.

The two render differently: `MEMBER_OF` is observed past, drawn solid; `INTERESTED_IN` is observed *intent* about an unrealised future, drawn dashed. A portal click must not look like four years of membership (NR-2a).

`signal_strength` is a word scale in the data, not a number, so the interface never has to render a club interest score (DR-6b).

Generated from the CSV by `data/write_schemas.py`; the examples are real rows, not illustrations.

| Column | Type | Example A | Example B |
| :--- | :--- | :--- | :--- |
| `person_id` | string | A26-101831 | L2023-10599 |
| `person_type` | enum | Applicant | Alum |
| `club_id` | string | CLB-05 | CLB-41 |
| `relationship` | enum | INTERESTED_IN | MEMBER_OF |
| `interest_signal` | enum, nullable | Essay Mention |  |
| `signal_strength` | enum, nullable | Passive |  |
| `club_role` | enum, nullable |  | President |
| `years_active` | integer, nullable |  | 3 |

## Enum values

- `person_type`: Alum, Applicant
- `relationship`: INTERESTED_IN, MEMBER_OF
- `interest_signal`: Application Checkbox, Campus Visit Meeting, Counsellor Note, Essay Mention, Info Session Topic, Portal Click
- `signal_strength`: Explicit, Passive, Sustained
- `club_role`: Founder, Member, Officer, President
