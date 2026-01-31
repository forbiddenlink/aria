"""Advanced memory system with episodic and semantic memory separation.

Based on 2026 best practices for AI agent memory architecture:
- Episodic Memory: Specific events and experiences (what happened, when, context)
- Semantic Memory: General knowledge and patterns (what I know, prefer, believe)
- Working Memory: Current context and active tasks
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class EpisodicMemory:
    """Memory of specific creative events and experiences."""

    def __init__(self):
        self.episodes: list[dict[str, Any]] = []

    def record_episode(
        self,
        event_type: str,
        details: dict[str, Any],
        emotional_state: dict[str, Any],
        timestamp: str | None = None,
    ):
        """Record a specific episode in Aria's creative journey."""
        episode = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "event_type": event_type,  # "creation", "reflection", "discovery", etc.
            "details": details,
            "emotional_state": emotional_state,
        }
        self.episodes.append(episode)
        logger.debug("episode_recorded", event_type=event_type)

    def get_recent_episodes(
        self, count: int = 10, event_type: str | None = None
    ) -> list[dict]:
        """Retrieve recent episodes, optionally filtered by type."""
        filtered = (
            self.episodes
            if not event_type
            else [ep for ep in self.episodes if ep["event_type"] == event_type]
        )
        return filtered[-count:]

    def get_episodes_by_mood(self, mood: str) -> list[dict]:
        """Get all episodes that occurred in a specific mood."""
        return [ep for ep in self.episodes if ep["emotional_state"].get("mood") == mood]

    def count_episodes_by_type(self) -> dict[str, int]:
        """Get statistics on episode types."""
        counts: dict[str, int] = defaultdict(int)
        for ep in self.episodes:
            counts[ep["event_type"]] += 1
        return dict(counts)


class SemanticMemory:
    """General knowledge and learned patterns about art and creativity."""

    def __init__(self):
        self.knowledge: dict[str, Any] = {
            "style_effectiveness": {},  # Which styles work well
            "subject_resonance": {},  # Which subjects resonate
            "mood_patterns": {},  # Patterns about mood influences
            "color_harmony": {},  # Color combinations that work
            "composition_rules": {},  # Compositional insights
            "learned_associations": [],  # General creative insights
        }

    def record_style_effectiveness(
        self, style: str, avg_score: float, sample_size: int
    ):
        """Learn which styles tend to produce better work."""
        self.knowledge["style_effectiveness"][style] = {
            "avg_score": avg_score,
            "sample_size": sample_size,
            "last_updated": datetime.now().isoformat(),
        }

    def record_subject_resonance(self, subject: str, metrics: dict[str, float]):
        """Learn which subjects resonate emotionally."""
        self.knowledge["subject_resonance"][subject] = {
            **metrics,
            "last_updated": datetime.now().isoformat(),
        }

    def learn_association(self, insight: str, category: str = "general"):
        """Record a general creative insight or learned association."""
        self.knowledge["learned_associations"].append(
            {
                "insight": insight,
                "category": category,
                "learned_at": datetime.now().isoformat(),
            }
        )

    def get_best_styles(self, min_samples: int = 3) -> list[tuple[str, float]]:
        """Get most effective styles based on learned patterns."""
        styles = [
            (style, data["avg_score"])
            for style, data in self.knowledge["style_effectiveness"].items()
            if data["sample_size"] >= min_samples
        ]
        return sorted(styles, key=lambda x: x[1], reverse=True)

    def get_insights_by_category(self, category: str) -> list[str]:
        """Retrieve learned insights by category."""
        return [
            assoc["insight"]
            for assoc in self.knowledge["learned_associations"]
            if assoc["category"] == category
        ]


class WorkingMemory:
    """Short-term memory for current creative session."""

    def __init__(self):
        self.current_context: dict[str, Any] = {}
        self.active_goals: list[str] = []
        self.session_start: str = datetime.now().isoformat()

    def set_context(self, key: str, value: Any):
        """Store information in working memory."""
        self.current_context[key] = value

    def get_context(self, key: str) -> Any:
        """Retrieve from working memory."""
        return self.current_context.get(key)

    def add_goal(self, goal: str):
        """Add an active creative goal."""
        self.active_goals.append(goal)

    def complete_goal(self, goal: str):
        """Mark a goal as completed."""
        if goal in self.active_goals:
            self.active_goals.remove(goal)

    def clear_session(self):
        """Clear working memory for new session."""
        self.current_context = {}
        self.active_goals = []
        self.session_start = datetime.now().isoformat()


class EnhancedMemorySystem:
    """Integrated memory system with episodic, semantic, and working memory.

    Based on 2026 AI agent memory architecture best practices.
    """

    def __init__(self, memory_file: Path = Path("data/aria_enhanced_memory.json")):
        self.memory_file = memory_file

        # Three types of memory
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.working = WorkingMemory()

        # Metadata
        self.created_at = datetime.now().isoformat()

        # Load existing memory
        self._load()

        logger.info(
            "enhanced_memory_initialized",
            memory_file=str(memory_file),
            episodes=len(self.episodic.episodes),
        )

    def _load(self):
        """Load memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file) as f:
                    data = json.load(f)

                self.episodic.episodes = data.get("episodic", {}).get("episodes", [])
                self.semantic.knowledge = data.get("semantic", {}).get(
                    "knowledge", self.semantic.knowledge
                )
                self.created_at = data.get("created_at", self.created_at)

                logger.info(
                    "enhanced_memory_loaded",
                    episodes=len(self.episodic.episodes),
                    styles_learned=len(self.semantic.knowledge["style_effectiveness"]),
                )
            except Exception as e:
                logger.error("enhanced_memory_load_failed", error=str(e))

    def save(self):
        """Persist memory to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "created_at": self.created_at,
                "last_saved": datetime.now().isoformat(),
                "episodic": {
                    "episodes": self.episodic.episodes,
                },
                "semantic": {
                    "knowledge": self.semantic.knowledge,
                },
                # Working memory is not persisted (session-specific)
            }

            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("enhanced_memory_saved", episodes=len(self.episodic.episodes))
        except Exception as e:
            logger.error("enhanced_memory_save_failed", error=str(e))

    def record_creation(
        self,
        artwork_details: dict[str, Any],
        emotional_state: dict[str, Any],
        outcome: dict[str, Any],
    ):
        """Record a creation event in episodic memory and update semantic knowledge."""
        # Episodic: What happened
        self.episodic.record_episode(
            event_type="creation",
            details=artwork_details,
            emotional_state=emotional_state,
        )

        # Semantic: Learn from it
        style = artwork_details.get("style", "unknown")
        score = outcome.get("score", 0.0)

        # Update style effectiveness
        if style != "unknown":
            current = self.semantic.knowledge["style_effectiveness"].get(
                style,
                {
                    "avg_score": 0.0,
                    "sample_size": 0,
                },
            )

            new_sample_size = current["sample_size"] + 1
            new_avg = (
                current["avg_score"] * current["sample_size"] + score
            ) / new_sample_size

            self.semantic.record_style_effectiveness(style, new_avg, new_sample_size)

        self.save()

    def generate_insights(self) -> dict[str, Any]:
        """Generate insights from accumulated memory."""
        insights: dict[str, Any] = {
            "total_creations": 0,
            "style_effectiveness": {},
            "mood_patterns": {},
            "best_performing_mood": None,
            "insights_text": [],
        }

        # Count creations
        episode_counts = self.episodic.count_episodes_by_type()
        total_creations = episode_counts.get("creation", 0)
        insights["total_creations"] = total_creations

        # Style effectiveness
        for style, data in self.semantic.knowledge["style_effectiveness"].items():
            if data["sample_size"] > 0:
                insights["style_effectiveness"][style] = data["avg_score"]

        # Mood patterns
        for episode in self.episodic.episodes:
            if episode["event_type"] == "creation":
                mood = episode["emotional_state"].get("mood", "unknown")
                insights["mood_patterns"][mood] = (
                    insights["mood_patterns"].get(mood, 0) + 1
                )

        # Best performing mood
        mood_scores: dict[str, list[float]] = {}
        for episode in self.episodic.episodes:
            if episode["event_type"] == "creation":
                mood = episode["emotional_state"].get("mood", "unknown")
                score = episode["details"].get("score", 0)
                if mood not in mood_scores:
                    mood_scores[mood] = []
                mood_scores[mood].append(score)

        if mood_scores:
            best_mood = None
            best_avg: float = -1.0
            for mood, scores in mood_scores.items():
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_mood = mood
            if best_mood:
                insights["best_performing_mood"] = {
                    "mood": best_mood,
                    "avg_score": best_avg,
                    "count": len(mood_scores[best_mood]),
                }

        # Generate text insights
        best_styles = self.semantic.get_best_styles(min_samples=2)
        if best_styles:
            top_style, top_score = best_styles[0]
            insights["insights_text"].append(
                f"I've found that {top_style} style consistently produces my best work (avg score: {top_score:.2f})"
            )

        if total_creations > 0:
            insights["insights_text"].append(
                f"I've created {total_creations} pieces so far in my journey"
            )

        return insights

    def get_relevant_context(self, current_mood: str, limit: int = 5) -> dict[str, Any]:
        """Retrieve relevant memories for current creative context."""
        return {
            "similar_mood_episodes": self.episodic.get_episodes_by_mood(current_mood)[
                -limit:
            ],
            "best_styles": self.semantic.get_best_styles()[:3],
            "recent_insights": self.semantic.knowledge["learned_associations"][-limit:],
        }

    def get_evolution_timeline(self) -> dict[str, Any]:
        """Get evolution timeline showing artistic growth over time.

        Returns data for Phase 5 Evolution Display:
        - Timeline of creations grouped by date
        - Style preference evolution
        - Mood distribution over time
        - Milestone creations
        """
        timeline: dict[str, Any] = {
            "phases": [],  # Artistic phases
            "milestones": [],  # Notable creations
            "style_evolution": [],  # How style preferences changed
            "mood_distribution": {},  # Mood counts by period
            "score_trend": [],  # Quality scores over time
        }

        if not self.episodic.episodes:
            return timeline

        # Group creations by date
        creations_by_date: dict[str, list[dict]] = defaultdict(list)
        for ep in self.episodic.episodes:
            if ep["event_type"] == "creation":
                # Extract date from timestamp
                ts = ep.get("timestamp", "")
                date = ts[:10] if len(ts) >= 10 else "unknown"
                creations_by_date[date].append(ep)

        # Build timeline with aggregated data
        sorted_dates = sorted(creations_by_date.keys())
        running_styles: dict[str, int] = defaultdict(int)

        for date in sorted_dates:
            day_creations = creations_by_date[date]
            day_scores = []
            day_moods: dict[str, int] = defaultdict(int)
            day_styles: dict[str, int] = defaultdict(int)

            for creation in day_creations:
                # Extract score
                score = creation.get("details", {}).get("score", 0)
                if score == 0:
                    # Try alternate location
                    score = creation.get("outcome", {}).get("score", 0)
                day_scores.append(score)

                # Extract mood
                mood = creation.get("emotional_state", {}).get("mood", "unknown")
                day_moods[mood] += 1

                # Extract style
                style = creation.get("details", {}).get("style", "unknown")
                day_styles[style] += 1
                running_styles[style] += 1

            # Add to timeline
            avg_score = sum(day_scores) / len(day_scores) if day_scores else 0
            timeline["score_trend"].append(
                {
                    "date": date,
                    "avg_score": round(avg_score, 3),
                    "count": len(day_creations),
                }
            )

            # Update mood distribution
            for mood, count in day_moods.items():
                timeline["mood_distribution"][mood] = (
                    timeline["mood_distribution"].get(mood, 0) + count
                )

            # Track style evolution
            if day_styles:
                dominant_style = max(day_styles, key=day_styles.get)  # type: ignore[arg-type]
                timeline["style_evolution"].append(
                    {
                        "date": date,
                        "dominant_style": dominant_style,
                        "styles_used": dict(day_styles),
                    }
                )

        # Identify milestones (high scores, first of each style, etc.)
        all_creations = [
            ep for ep in self.episodic.episodes if ep["event_type"] == "creation"
        ]
        if all_creations:
            # Best creation
            best = max(
                all_creations,
                key=lambda x: x.get("details", {}).get("score", 0),
            )
            best_score = best.get("details", {}).get("score", 0)
            if best_score > 0:
                timeline["milestones"].append(
                    {
                        "type": "best_creation",
                        "date": best.get("timestamp", "")[:10],
                        "description": f"Highest quality creation (score: {best_score:.2f})",
                        "details": {
                            "style": best.get("details", {}).get("style"),
                            "mood": best.get("emotional_state", {}).get("mood"),
                        },
                    }
                )

            # First creation
            first = all_creations[0]
            timeline["milestones"].append(
                {
                    "type": "first_creation",
                    "date": first.get("timestamp", "")[:10],
                    "description": "First artwork created",
                    "details": {
                        "style": first.get("details", {}).get("style"),
                        "mood": first.get("emotional_state", {}).get("mood"),
                    },
                }
            )

            # Style discoveries (first use of each style)
            seen_styles: set[str] = set()
            for creation in all_creations:
                style = creation.get("details", {}).get("style", "unknown")
                if style != "unknown" and style not in seen_styles:
                    seen_styles.add(style)
                    if len(seen_styles) > 1:  # Skip the very first
                        timeline["milestones"].append(
                            {
                                "type": "style_discovery",
                                "date": creation.get("timestamp", "")[:10],
                                "description": f"First experimented with {style} style",
                                "details": {"style": style},
                            }
                        )

        # Identify artistic phases (clusters of similar moods/styles)
        if len(sorted_dates) >= 3:
            # Simple phase detection: group consecutive days with same dominant mood
            current_phase_mood = None
            phase_start = None
            phases: list[dict] = []

            for entry in timeline["score_trend"]:
                date = entry["date"]
                day_creations = creations_by_date.get(date, [])
                if day_creations:
                    moods = [
                        c.get("emotional_state", {}).get("mood") for c in day_creations
                    ]
                    dominant_mood = max(set(moods), key=moods.count) if moods else None

                    if dominant_mood != current_phase_mood:
                        if current_phase_mood and phase_start:
                            phases.append(
                                {
                                    "mood": current_phase_mood,
                                    "start_date": phase_start,
                                    "end_date": date,
                                    "name": f"{current_phase_mood.title()} Period",
                                }
                            )
                        current_phase_mood = dominant_mood
                        phase_start = date

            # Close final phase
            if current_phase_mood and phase_start:
                phases.append(
                    {
                        "mood": current_phase_mood,
                        "start_date": phase_start,
                        "end_date": sorted_dates[-1] if sorted_dates else phase_start,
                        "name": f"{current_phase_mood.title()} Period",
                    }
                )

            timeline["phases"] = phases

        return timeline

    def get_style_preferences_over_time(self) -> list[dict[str, Any]]:
        """Track how style preferences evolved over time."""
        style_by_month: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for ep in self.episodic.episodes:
            if ep["event_type"] == "creation":
                ts = ep.get("timestamp", "")
                month = ts[:7] if len(ts) >= 7 else "unknown"  # YYYY-MM
                style = ep.get("details", {}).get("style", "unknown")
                score = ep.get("details", {}).get("score", 0.5)

                # Weight by score - better creations influence preferences more
                style_by_month[month][style] += score

        # Convert to sorted list
        result = []
        for month in sorted(style_by_month.keys()):
            styles = style_by_month[month]
            if styles:
                total = sum(styles.values())
                preferences = {s: round(v / total, 2) for s, v in styles.items()}
                result.append(
                    {
                        "month": month,
                        "preferences": preferences,
                        "dominant": max(preferences, key=preferences.get),  # type: ignore[arg-type]
                    }
                )

        return result
