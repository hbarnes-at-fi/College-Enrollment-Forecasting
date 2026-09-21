# Schema: `campus_clubs_dim.csv`

45 rows x 10 columns. The DR-6 campus club roster. Distinct from activities: these are organisations that exist here, each with a headcount and a viability floor.

`linked_activity_id` is the bridge that connects a high school activity to its campus counterpart, so HS Debate reaches the Debate Society. Seven clubs deliberately have none; a bridge that always exists carries no information.

`proxy_sensitive = Y` marks affinity and religious organisations. These are included in succession projection, roster panels, and the graph, but **excluded from similarity scoring and from P3 coverage credit** (specs.md section 0.1). Interest in one is a proxy for race or religion, which SR-3 excludes.

A club is at succession risk when `current_members - graduating_members + expected new members` falls below `min_viable_members`. Nine of 45 are, per DR-6c.

Generated from the CSV by `data/write_schemas.py`; the examples are real rows, not illustrations.

| Column | Type | Example A | Example B |
| :--- | :--- | :--- | :--- |
| `club_id` | string | CLB-01 | CLB-02 |
| `club_name` | string | Debate Society | Model United Nations |
| `category` | enum | Competitive intellectual | Competitive intellectual |
| `linked_activity_id` | string, nullable | ACT-01 | ACT-02 |
| `current_members` | integer | 14 | 20 |
| `graduating_members` | integer | 2 | 5 |
| `min_viable_members` | integer | 8 | 10 |
| `charter_status` | enum | Active | Active |
| `advisor_backed` | Y/N | Y | Y |
| `proxy_sensitive` | Y/N | N | N |

## Enum values

- `category`: Arts, Competitive intellectual, Enterprise, Governance, Identity and culture, Publications, Research, Service
- `charter_status`: Active, Probationary
- `advisor_backed`: N, Y
- `proxy_sensitive`: N, Y
