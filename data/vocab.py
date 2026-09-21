"""Static vocabularies: the source of truth for DR-3 and DR-6 identity columns.

specs.md sections 2 and 3. The generator writes these to CSV and fills the
numeric roster columns; identity lives here so it stays reviewable.
"""

# ----------------------------------------------------- DR-3, specs §2
# (activity_id, activity_name, category)
NON_ATHLETIC = [
    ("ACT-01", "Debate and Speech", "Competitive intellectual"),
    ("ACT-02", "Model UN", "Competitive intellectual"),
    ("ACT-03", "Math Olympiad", "Competitive intellectual"),
    ("ACT-04", "Science Olympiad", "Competitive intellectual"),
    ("ACT-05", "Robotics", "Competitive intellectual"),
    ("ACT-06", "Quiz Bowl", "Competitive intellectual"),
    ("ACT-07", "Instrumental Music", "Arts"),
    ("ACT-08", "Vocal Music", "Arts"),
    ("ACT-09", "Theatre", "Arts"),
    ("ACT-10", "Dance", "Arts"),
    ("ACT-11", "Studio Art", "Arts"),
    ("ACT-12", "Film", "Arts"),
    ("ACT-13", "Newspaper", "Publications"),
    ("ACT-14", "Yearbook", "Publications"),
    ("ACT-15", "Literary Magazine", "Publications"),
    ("ACT-16", "Radio", "Publications"),
    ("ACT-17", "Student Government", "Governance"),
    ("ACT-18", "Class Officer", "Governance"),
    ("ACT-19", "Honour Council", "Governance"),
    ("ACT-20", "Community Volunteering", "Service"),
    ("ACT-21", "Tutoring and Peer Mentoring", "Service"),
    ("ACT-22", "Cultural and Affinity Organisations", "Identity and culture"),
    ("ACT-23", "International Student Association", "Identity and culture"),
    ("ACT-24", "Entrepreneurship", "Enterprise"),
    ("ACT-25", "Business and Investment", "Enterprise"),
    ("ACT-26", "DECA", "Enterprise"),
    ("ACT-27", "Independent Research", "Research"),
    ("ACT-28", "Published Work", "Research"),
    ("ACT-29", "Lab Assistantship", "Research"),
    ("ACT-30", "Part-time Employment", "Obligation"),
    ("ACT-31", "Family Caregiving", "Obligation"),
]

# Names match `athletic_recruit_sport` in mock_admit_pool_fall2026.csv exactly,
# so DR-1 athletics join without a mapping table.
SPORTS = [
    "Football", "Field Hockey", "Men's Soccer", "Women's Soccer",
    "Women's Lacrosse", "Men's Lacrosse", "Baseball", "Softball",
    "Swimming & Diving", "Track & Field", "Women's Basketball",
    "Men's Basketball", "Volleyball", "Rowing", "Golf", "Wrestling",
]
ATHLETIC = [(f"ACT-{32 + i}", name, "Athletics") for i, name in enumerate(SPORTS)]

ACTIVITIES = NON_ATHLETIC + ATHLETIC          # 31 + 16 = 47
SPORT_TO_ACT = {name: aid for aid, name, _ in ATHLETIC}

# Activities that can plausibly be pursued in high school. Obligation and
# Research still count; Class Officer is school-specific but portable.
HS_ELIGIBLE = [a[0] for a in NON_ATHLETIC]

# ----------------------------------------------------- DR-6, specs §3
# (club_id, club_name, category, linked_activity_id|None,
#  proxy_sensitive, at_risk_required)
CLUBS = [
    ("CLB-01", "Debate Society", "Competitive intellectual", "ACT-01", False, False),
    ("CLB-02", "Model United Nations", "Competitive intellectual", "ACT-02", False, False),
    ("CLB-03", "Math Club", "Competitive intellectual", "ACT-03", False, False),
    ("CLB-04", "Science Olympiad Team", "Competitive intellectual", "ACT-04", False, False),
    ("CLB-05", "Robotics Team", "Competitive intellectual", "ACT-05", False, False),
    ("CLB-06", "Quiz Bowl", "Competitive intellectual", "ACT-06", False, True),
    ("CLB-07", "Concert Band", "Arts", "ACT-07", False, False),
    ("CLB-08", "Orchestra", "Arts", "ACT-07", False, False),
    ("CLB-09", "Chamber Choir", "Arts", "ACT-08", False, False),
    ("CLB-10", "A Cappella Ensemble", "Arts", "ACT-08", False, False),
    ("CLB-11", "Student Theatre Company", "Arts", "ACT-09", False, False),
    ("CLB-12", "Dance Collective", "Arts", "ACT-10", False, False),
    ("CLB-13", "Studio Art Collective", "Arts", "ACT-11", False, False),
    ("CLB-14", "Film Society", "Arts", "ACT-12", False, False),
    ("CLB-15", "Student Newspaper", "Publications", "ACT-13", False, False),
    ("CLB-16", "Yearbook", "Publications", "ACT-14", False, True),
    ("CLB-17", "Literary Review", "Publications", "ACT-15", False, False),
    ("CLB-18", "Campus Radio", "Publications", "ACT-16", False, True),
    ("CLB-19", "Student Government Association", "Governance", "ACT-17", False, False),
    ("CLB-20", "Honour Council", "Governance", "ACT-19", False, True),
    ("CLB-21", "Volunteer Corps", "Service", "ACT-20", False, False),
    ("CLB-22", "Peer Tutoring Centre", "Service", "ACT-21", False, False),
    ("CLB-23", "International Students Association", "Identity and culture", "ACT-23", False, False),
    ("CLB-24", "Entrepreneurship Society", "Enterprise", "ACT-24", False, False),
    ("CLB-25", "Investment Club", "Enterprise", "ACT-25", False, False),
    ("CLB-26", "Undergraduate Research Collective", "Research", "ACT-27", False, False),
    # proxy_sensitive: rendered and projected, but excluded from similarity
    # scoring and from P3 coverage credit. specs.md section 0.1.
    ("CLB-27", "Black Student Union", "Identity and culture", "ACT-22", True, False),
    ("CLB-28", "Latino Student Alliance", "Identity and culture", "ACT-22", True, False),
    ("CLB-29", "Asian Students Association", "Identity and culture", "ACT-22", True, False),
    ("CLB-30", "Hillel", "Identity and culture", "ACT-22", True, True),
    ("CLB-31", "Newman Catholic Community", "Identity and culture", "ACT-22", True, False),
    ("CLB-32", "Muslim Students Association", "Identity and culture", "ACT-22", True, False),
    ("CLB-33", "Outdoors Club", "Service", None, False, False),
    ("CLB-34", "Chess Club", "Competitive intellectual", None, False, True),
    ("CLB-35", "Esports Association", "Competitive intellectual", None, False, False),
    ("CLB-36", "Sustainability Coalition", "Service", None, False, False),
    ("CLB-37", "Habitat Build Chapter", "Service", "ACT-20", False, False),
    ("CLB-38", "Pre-Health Society", "Research", None, False, False),
    ("CLB-39", "Mock Trial", "Competitive intellectual", "ACT-01", False, False),
    ("CLB-40", "Political Union", "Governance", "ACT-17", False, False),
    ("CLB-41", "Photography Club", "Arts", "ACT-11", False, False),
    ("CLB-42", "Culinary Society", "Enterprise", None, False, True),
    ("CLB-43", "Board Game Guild", "Competitive intellectual", None, False, True),
    ("CLB-44", "Ballroom Dance Society", "Arts", "ACT-10", False, False),
    ("CLB-45", "Astronomy Club", "Research", "ACT-04", False, True),
]

CLUB_IDS = [c[0] for c in CLUBS]
PROXY_SENSITIVE = {c[0] for c in CLUBS if c[4]}
AT_RISK_REQUIRED = {c[0] for c in CLUBS if c[5]}
LINKED = {c[0]: c[3] for c in CLUBS}

# ----------------------------------------------------- distributions, specs §4
ROLE_DIST = {"Member": 0.62, "Officer": 0.20, "Captain": 0.08,
             "President": 0.06, "Founder": 0.04}
RECOG_DIST = {"None": 0.50, "School": 0.24, "Regional": 0.14,
              "State": 0.09, "National": 0.03}
SIGNAL_DIST = {"Portal Click": 0.31, "Info Session Topic": 0.18,
               "Campus Visit Meeting": 0.12, "Essay Mention": 0.15,
               "Application Checkbox": 0.19, "Counsellor Note": 0.05}
STRENGTH_DIST = {"Passive": 0.46, "Explicit": 0.38, "Sustained": 0.16}
CLUB_ROLE_DIST = {"Member": 0.68, "Officer": 0.19,
                  "President": 0.08, "Founder": 0.05}

# Amended from the 1.6 in specs §4.1. At 1.6 the top activity is shared by 94%
# of people, which dominates the layout and carries no information. At 0.8 the
# spread runs 60% down to 5%, a ~6x idf range, so SR-4 rarity weighting still
# discriminates without a degenerate hub.
ZIPF_EXPONENT = 0.8
ACT_EDGES_MIN, ACT_EDGES_MAX = 3, 6
CLUB_INTEREST_MEAN = 1.8
CLUB_MEMBER_MEAN = 2.1
NO_CLUB_INTEREST_SHARE = 0.14
BRIDGE_ASSOCIATION = 0.45       # DR-6d target, Cramer's V
