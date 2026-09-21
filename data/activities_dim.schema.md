# Schema: `activities_dim.csv`

47 rows x 3 columns. The DR-3 controlled vocabulary. One shared namespace across both populations; without it nothing links.

31 non-athletic activities plus 16 sports. Sport names match `athletic_recruit_sport` in the admit pool exactly, so DR-1 athletics join without a mapping table. Sports are rendered as the Athletic team node type, not as Activity, per specs.md section 0.2 — one node per sport, not two.

Generated from the CSV by `data/write_schemas.py`; the examples are real rows, not illustrations.

| Column | Type | Example A | Example B |
| :--- | :--- | :--- | :--- |
| `activity_id` | string | ACT-01 | ACT-02 |
| `activity_name` | string | Debate and Speech | Model UN |
| `category` | enum | Competitive intellectual | Competitive intellectual |

## Enum values

- `category`: Arts, Athletics, Competitive intellectual, Enterprise, Governance, Identity and culture, Obligation, Publications, Research, Service
