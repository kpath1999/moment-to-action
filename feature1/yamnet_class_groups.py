"""
yamnet_class_groups.py
----------------------
Semantic grouping of all 521 YAMNet AudioSet classes into a context-aware
scoring framework for security/violence detection.

Design philosophy:
  - Every class is assigned to exactly one group (or IGNORE)
  - Groups feed into three layers: Environment, Scene, Event
  - Environment and Scene modulate the threat prior
  - Event groups produce the raw trigger signal
  - No training required — weights are tunable by observation

Usage:
    from yamnet_class_groups import ClassGrouper, score_threat

    grouper = ClassGrouper(labels)          # pass your loaded labels dict {idx: name}
    group_scores = grouper.aggregate(yamnet_scores)
    threat = score_threat(group_scores)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GROUP DEFINITIONS
# Each entry: class name substring (case-insensitive match) → group name
# First match wins — order within each section matters for ambiguous names.
# ─────────────────────────────────────────────────────────────────────────────

CLASS_TO_GROUP: list[tuple[str, str]] = [

    # ── LAYER 1: ENVIRONMENT ─────────────────────────────────────────────────
    # Tells us WHERE we are. Modulates all downstream thresholds.

    # Outdoor indicators
    ("Wind",                        "env_outdoor"),
    ("Rain",                        "env_outdoor"),
    ("Thunder",                     "env_outdoor"),
    ("Bird",                        "env_outdoor"),
    ("Cricket",                     "env_outdoor"),
    ("Frog",                        "env_outdoor"),
    ("Insect",                      "env_outdoor"),
    ("Ocean",                       "env_outdoor"),
    ("Stream",                      "env_outdoor"),
    ("Waterfall",                   "env_outdoor"),
    ("Traffic noise",               "env_outdoor"),
    ("Road",                        "env_outdoor"),

    # Indoor indicators
    ("Echo",                        "env_indoor"),
    ("Reverberation",               "env_indoor"),
    ("Air conditioning",            "env_indoor"),
    ("Mechanical fan",              "env_indoor"),
    ("Toilet flush",                "env_indoor"),
    ("Sink",                        "env_indoor"),
    ("Microwave oven",              "env_indoor"),
    ("Vacuum cleaner",              "env_indoor"),
    ("Clock",                       "env_indoor"),
    ("Printer",                     "env_indoor"),
    ("Computer keyboard",           "env_indoor"),
    ("Typing",                      "env_indoor"),
    ("Door",                        "env_indoor"),

    # Crowd density
    ("Crowd",                       "env_crowd"),
    ("Hubbub",                      "env_crowd"),
    ("Chatter",                     "env_crowd"),
    ("Babble",                      "env_crowd"),
    ("Applause",                    "env_crowd"),
    ("Cheering",                    "env_crowd"),
    ("Booing",                      "env_crowd"),
    ("Chant",                       "env_crowd"),

    # ── LAYER 2: SCENE CONTEXT ───────────────────────────────────────────────
    # Tells us WHAT SITUATION we're in.
    # High scores here suppress the threat prior significantly.

    # Entertainment / sports — screaming here is almost never a threat
    ("Music",                       "scene_entertainment"),
    ("Singing",                     "scene_entertainment"),
    ("Song",                        "scene_entertainment"),
    ("Choir",                       "scene_entertainment"),
    ("Cheer",                       "scene_entertainment"),
    ("Sports commentary",           "scene_entertainment"),
    ("Crowd chant",                 "scene_entertainment"),
    ("Stadium",                     "scene_entertainment"),
    ("Concert",                     "scene_entertainment"),
    ("DJ",                          "scene_entertainment"),
    ("Electronic music",            "scene_entertainment"),
    ("Hip hop",                     "scene_entertainment"),
    ("Rock music",                  "scene_entertainment"),
    ("Pop music",                   "scene_entertainment"),
    ("Drum",                        "scene_entertainment"),
    ("Bass guitar",                 "scene_entertainment"),
    ("Electric guitar",             "scene_entertainment"),
    ("Trumpet",                     "scene_entertainment"),
    ("Violin",                      "scene_entertainment"),
    ("Piano",                       "scene_entertainment"),
    ("Synthesizer",                 "scene_entertainment"),
    ("Beatboxing",                  "scene_entertainment"),
    ("Rapping",                     "scene_entertainment"),

    # Domestic / home — reduces outdoor threat model
    ("Television",                  "scene_domestic"),
    ("Radio",                       "scene_domestic"),
    ("Baby",                        "scene_domestic"),
    ("Cat",                         "scene_domestic"),
    ("Dog",                         "scene_domestic"),
    ("Cooking",                     "scene_domestic"),
    ("Dishes, pots, and pans",      "scene_domestic"),
    ("Cutlery",                     "scene_domestic"),
    ("Chopping",                    "scene_domestic"),
    ("Frying",                      "scene_domestic"),
    ("Microwave",                   "scene_domestic"),

    # Transport context — vehicle sounds anchor scene
    ("Car",                         "scene_transport"),
    ("Truck",                       "scene_transport"),
    ("Bus",                         "scene_transport"),
    ("Motorcycle",                  "scene_transport"),
    ("Bicycle",                     "scene_transport"),
    ("Train",                       "scene_transport"),
    ("Aircraft",                    "scene_transport"),
    ("Helicopter",                  "scene_transport"),
    ("Boat",                        "scene_transport"),
    ("Engine",                      "scene_transport"),
    ("Motor vehicle (road)",        "scene_transport"),
    ("Subway, metro, underground",  "scene_transport"),

    # Work / industrial — noisy but not threatening
    ("Power tool",                  "scene_industrial"),
    ("Drill",                       "scene_industrial"),
    ("Saw",                         "scene_industrial"),
    ("Jackhammer",                  "scene_industrial"),
    ("Hammer",                      "scene_industrial"),
    ("Construction",                "scene_industrial"),
    ("Factory",                     "scene_industrial"),
    ("Machine",                     "scene_industrial"),
    ("Chainsaw",                    "scene_industrial"),

    # ── FALSE POSITIVE SUPPRESSORS ───────────────────────────────────────────
    # High energy / expressive speech that is NOT distress.
    # Strong signal here should veto or heavily discount a distress trigger.

    ("Laughter",                    "suppress_fp"),
    ("Giggle",                      "suppress_fp"),
    ("Chuckle",                     "suppress_fp"),
    ("Belly laugh",                 "suppress_fp"),
    ("Snicker",                     "suppress_fp"),
    ("Children playing",            "suppress_fp"),
    ("Child speech",                "suppress_fp"),     # excited kids, not distress
    ("Narration, monologue",        "suppress_fp"),
    ("Speech synthesizer",          "suppress_fp"),
    ("Conversation",                "suppress_fp"),

    # ── LAYER 3: EVENT DETECTION — DISTRESS VOCALS ───────────────────────────
    # Core trigger classes. Fused with acoustic gate score.

    ("Screaming",                   "vocal_distress"),
    ("Scream",                      "vocal_distress"),
    ("Yell",                        "vocal_distress"),
    ("Shout",                       "vocal_distress"),
    ("Bellow",                      "vocal_distress"),
    ("Whoop",                       "vocal_distress"),
    ("Children shouting",           "vocal_distress"),
    ("Crying, sobbing",             "vocal_distress"),
    ("Wail, moan",                  "vocal_distress"),
    ("Whimper",                     "vocal_distress"),
    ("Groan",                       "vocal_distress"),

    # General speech — ambiguous, used to detect "someone is here"
    # but not a trigger on its own
    ("Speech",                      "vocal_speech"),
    ("Babbling",                    "vocal_speech"),
    ("Whispering",                  "vocal_speech"),

    # ── LAYER 3: EVENT DETECTION — PHYSICAL IMPACT ───────────────────────────
    # Co-occurring with vocal distress, these strongly confirm violence.
    # On their own, moderate evidence.

    ("Slap, smack",                 "impact_body"),
    ("Thud",                        "impact_body"),
    ("Grunt",                       "impact_body"),
    ("Grunting",                    "impact_body"),
    ##check breathing (is it heavy breathing, if i'm too close to the mic, it might be a FP)
    ("Breathing",                   "impact_body"),    # heavy breathing / struggle
    ("Pant",                        "impact_body"),

    ("Bang",                        "impact_object"),
    ("Crash",                       "impact_object"),
    ("Smash",                       "impact_object"),
    ("Breaking",                    "impact_object"),
    ("Shatter",                     "impact_object"),
    ("Slam",                        "impact_object"),
    ("Thump, thud",                 "impact_object"),
    ("Knock",                       "impact_object"),

    # ── LAYER 3: EVENT DETECTION — WEAPONS / CRITICAL ────────────────────────
    # High specificity. Treated as near-certain trigger regardless of context.

    ("Gunshot, gunfire",            "weapon"),
    ("Gunfire",                     "weapon"),
    ("Gunshot",                     "weapon"),
    ("Explosion",                   "weapon"),
    ("Burst, pop",                  "weapon"),          # can be gunfire
    ("Glass",                       "weapon"),          # breaking glass
    ("Screech of tires",            "weapon"),          # getaway vehicle
    ("Siren",                       "weapon"),          # police/ambulance context

    # ── LAYER 3: EVENT DETECTION — THREAT CONTEXT ────────────────────────────
    # Not triggers alone, but raise the prior when combined with vocal distress.

    ("Riot",                        "threat_context"),
    ("Crowd chanting",              "threat_context"),  # aggressive, not entertainment
    ("Booing",                      "threat_context"),
    ("Angry",                       "threat_context"),
    ("Aggressive",                  "threat_context"),

    # ── IGNORE — no signal value for security detection ───────────────────────
    # Everything not matched above falls through to IGNORE automatically.
    # Listed here explicitly for documentation only.

    # Animals (non-threat)
    ("Animal",                      "ignore"),
    ("Livestock",                   "ignore"),
    ("Wild animals",                "ignore"),
    ("Domestic animals",            "ignore"),

    # Nature (non-threat)
    ("Fire",                        "ignore"),          # ambiguous — could keep
    ("Water",                       "ignore"),
    ("Silence",                     "ignore"),

    # Music instruments (covered by scene_entertainment above, but belt+suspenders)
    ("Guitar",                      "ignore"),
    ("Flute",                       "ignore"),
    ("Saxophone",                   "ignore"),
    ("Cello",                       "ignore"),
    ("Accordion",                   "ignore"),

    # Misc
    ("Telephone",                   "ignore"),
    ("Bell",                        "ignore"),
    ("Alarm",                       "ignore"),          # could argue for threat_context
    ("Ringtone",                    "ignore"),
    ("Notification",                "ignore"),
]


# ─────────────────────────────────────────────────────────────────────────────
# THREAT SCORING CONFIG
# All active group weights sum to 1.0 (weapon overrides to 1.0 separately).
# Tune thresholds after observing --debug output on real audio.
# ─────────────────────────────────────────────────────────────────────────────

# Minimum group score to activate that group's full weight contribution.
# Below threshold → 0. Above → full weight. Tweak after seeing real scores.
GROUP_THRESHOLDS = {
    "vocal_distress": 0.05,
    "impact_body":    0.05,
    "impact_object":  0.05,
}

# Speech + distress co-occurrence thresholds
SPEECH_THRESHOLD   = 0.40   # vocal_speech must exceed this
DISTRESS_THRESHOLD = 0.01   # vocal_distress sum must exceed this

# Weapon bypass threshold — kept high, only unambiguous weapon classes should fire
WEAPON_THRESHOLD = 0.05

# Weights — all four sum to 1.0
# vocal_distress + impact_body + impact_object + shouted_speech = 1.0
GROUP_WEIGHTS = {
    "vocal_distress": 0.40,
    "impact_body":    0.20,
    "impact_object":  0.10,
}
SHOUTED_SPEECH_WEIGHT = 0.30   # activates when speech > 0.4 AND distress > 0.01


# ─────────────────────────────────────────────────────────────────────────────
# THREAT SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_threat(group_scores: dict[str, float]) -> dict:
    """
    Compute threat score as a clean percentage [0.0, 1.0].

    Weights sum to 1.0:
        vocal_distress  0.40  (active if group sum > 0.05)
        impact_body     0.20  (active if group sum > 0.05)
        impact_object   0.10  (active if group sum > 0.05)
        shouted_speech  0.30  (active if speech > 0.40 AND distress > 0.01)
        ─────────────────────
        max total       1.00

    Weapon fires above WEAPON_THRESHOLD → immediate 1.0, bypasses all groups.
    Final score capped at 1.0.
    """
    components = {}

    # ── Weapon bypass ─────────────────────────────────────────────────────────
    weapon_score     = group_scores.get("weapon", 0.0)
    weapon_triggered = weapon_score >= WEAPON_THRESHOLD
    components["weapon"] = weapon_score

    if weapon_triggered:
        return {
            "threat_score":     1.0,
            "weapon_triggered": True,
            "components":       components,
        }

    # ── Group activation ──────────────────────────────────────────────────────
    threat = 0.0
    for group, weight in GROUP_WEIGHTS.items():
        score        = group_scores.get(group, 0.0)
        active       = score >= GROUP_THRESHOLDS[group]
        contribution = weight if active else 0.0
        threat      += contribution
        components[group] = {
            "score":        score,
            "active":       active,
            "contribution": contribution,
        }

    # ── Shouted speech ────────────────────────────────────────────────────────
    speech   = group_scores.get("vocal_speech",   0.0)
    distress = group_scores.get("vocal_distress", 0.0)
    shouted_speech = 0.0
    if speech > SPEECH_THRESHOLD and distress > DISTRESS_THRESHOLD:
        shouted_speech = SHOUTED_SPEECH_WEIGHT
    threat += shouted_speech
    components["shouted_speech"] = {
        "speech":       speech,
        "distress":     distress,
        "contribution": shouted_speech,
    }

    return {
        "threat_score":     min(1.0, threat),
        "weapon_triggered": False,
        "components":       components,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_group_report(group_scores: dict[str, float],
                       threat_result: dict,
                       min_score: float = 0.05) -> None:
    """Print a human-readable breakdown per chunk for calibration."""
    print("\n── Group Scores ─────────────────────────────────────")
    for group, score in sorted(group_scores.items(), key=lambda x: -x[1]):
        if score >= min_score and group not in ("ignore", "unmatched"):
            bar = "█" * min(30, int(score * 30))
            print(f"  {group:30s}  {score:.3f}  {bar}")

    c = threat_result["components"]
    print("\n── Threat Breakdown ─────────────────────────────────")
    for group, weight in GROUP_WEIGHTS.items():
        info   = c.get(group, {})
        status = "ACTIVE  " if info.get("active") else "inactive"
        print(f"  {group:20s}  score={info.get('score', 0):.3f}  "
              f"{status}  contribution={info.get('contribution', 0):.0%}")

    ss = c.get("shouted_speech", {})
    status = "ACTIVE  " if ss.get("contribution", 0) > 0 else "inactive"
    print(f"  {'shouted_speech':20s}  speech={ss.get('speech', 0):.3f}  "
          f"distress={ss.get('distress', 0):.3f}  "
          f"{status}  contribution={ss.get('contribution', 0):.0%}")

    print(f"  {'weapon':20s}  score={c.get('weapon', 0):.3f}")
    print(f"\n  ► THREAT SCORE:  {threat_result['threat_score']*100:.1f}%"
          f"{'  ⚠️  WEAPON BYPASS' if threat_result['weapon_triggered'] else ''}")
    print("─────────────────────────────────────────────────────")

class ClassGrouper:
    """
    Maps YAMNet's 521 class scores to semantic group scores.
    Build once at startup, call aggregate() on every inference.
    """

    def __init__(self, labels: dict[int, str]):
        """
        Args:
            labels: {index: display_name} dict from your yamnet_class_map.csv
        """
        self.labels = labels
        self._index_to_group = self._build_index_map()
        self._all_groups = {g for _, g in CLASS_TO_GROUP} | {"ignore", "unmatched"}

    def _build_index_map(self) -> dict[int, str]:
        """Pre-compute index → group for O(1) lookup at inference time."""
        index_map = {}
        for idx, name in self.labels.items():
            group = "unmatched"
            for substring, grp in CLASS_TO_GROUP:
                if substring.lower() in name.lower():
                    group = grp
                    break
            index_map[idx] = group
        return index_map

    # Groups that accumulate via raw sum instead of max.
    # No clipping, no normalization — co-firing classes stack freely.
    # Normalize later once you have a reliable reference distribution
    # from real deployment audio (use empirical 95th percentile as ceiling).
    SUM_GROUPS = {"vocal_distress", "impact_body", "impact_object", "suppress_fp"}

    def aggregate(self, scores: np.ndarray) -> dict[str, float]:
        """
        Aggregate YAMNet scores into group scores.

        SUM_GROUPS  → raw sum  (co-firing distress classes accumulate freely)
        all others  → max      (one strong class is enough)
        """
        if len(scores.shape) > 1:
            scores = scores[0] if scores.shape[0] == 1 else np.mean(scores, axis=0)

        group_scores: dict[str, list[float]] = {g: [] for g in self._all_groups}

        for idx, score in enumerate(scores):
            group = self._index_to_group.get(idx, "unmatched")
            group_scores[group].append(float(score))

        result = {}
        for grp, vals in group_scores.items():
            if not vals:
                result[grp] = 0.0
            elif grp in self.SUM_GROUPS:
                result[grp] = sum(vals)   # raw sum — let co-firing classes accumulate freely
            else:
                result[grp] = max(vals)   # max — one strong class is enough
        return result

    def top_classes_in_group(self, scores: np.ndarray, group: str,
                              top_k: int = 3) -> list[tuple[str, float]]:
        """Debug helper: return top-k class names within a group."""
        matches = [
            (self.labels[idx], float(scores[idx]))
            for idx, grp in self._index_to_group.items()
            if grp == group
        ]
        return sorted(matches, key=lambda x: x[1], reverse=True)[:top_k]
