"""Aria's memory system - Journaling and reflection on her creative journey."""

import json
from datetime import datetime
from pathlib import Pathfrom typing import Any
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ArtistMemory:
    """Aria's memory of her creative journey.

    Stores:
    - Paintings created
    - Reflections on each piece
    - Evolution of preferences
    - Notable moments in her artistic development
    """

    def __init__(self, memory_file: Path = Path("data/aria_memory.json")):
        self.memory_file = memory_file
        self.memory = {
            "artist_name": "Aria",
            "started_creating": datetime.now().isoformat(),
            "paintings": [],
            "reflections": [],
            "milestones": [],
            "personality_snapshots": [],
            "total_creations": 0,
        }

        # Load existing memory if available
        self._load_memory()

        logger.info(
            "memory_initialized",
            memory_file=str(self.memory_file),
            total_creations=self.memory["total_creations"],
        )

    def _load_memory(self):
        """Load memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file) as f:
                    self.memory = json.load(f)
                logger.info(
                    "memory_loaded",
                    total_creations=self.memory["total_creations"],
                    paintings=len(self.memory.get("paintings", [])),
                )
            except Exception as e:
                logger.error("memory_load_failed", error=str(e))

    def save_memory(self):
        """Save memory to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
            logger.info("memory_saved", file=str(self.memory_file))
        except Exception as e:
            logger.error("memory_save_failed", error=str(e))

    def record_painting(
        self,
        image_path: str,
        prompt: str,
        mood: str,
        style: str,
        subject: str,
        score: float,
        reflection: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        """Record a newly created painting."""
        painting_record = {
            "number": int(self.memory.get("total_creations", 0)) + 1,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "prompt": prompt,
            "mood": mood,
            "style": style,
            "subject": subject,
            "score": score,
            "reflection": reflection,
            "metadata": metadata or {},
        }

        paintings = self.memory.get("paintings", [])
        if not isinstance(paintings, list):
            paintings = []
        paintings.append(painting_record)
        self.memory["paintings"] = paintings
        self.memory["total_creations"] = int(self.memory.get("total_creations", 0)) + 1

        logger.info(
            "painting_recorded",
            number=painting_record["number"],
            score=score,
            mood=mood,
        )

        self.save_memory()

    def record_reflection(
        self, reflection: str, context: dict[Any, Any] | None = None
    ) -> None:
        """Record a general reflection or thought."""
        reflection_entry = {
            "timestamp": datetime.now().isoformat(),
            "reflection": reflection,
            "context": context or {},
        }

        reflections = self.memory.get("reflections", [])
        if not isinstance(reflections, list):
            reflections = []
        reflections.append(reflection_entry)
        self.memory["reflections"] = reflections
        logger.debug("reflection_recorded", length=len(reflection))
        self.save_memory()

    def record_milestone(self, milestone: str, details: dict = None):
        """Record a significant milestone in artistic development."""
        milestone_entry = {
            "timestamp": datetime.now().isoformat(),
            "milestone": milestone,
            "creation_count": self.memory["total_creations"],
            "details": details or {},
        }

        self.memory["milestones"].append(milestone_entry)
        logger.info("milestone_recorded", milestone=milestone)
        self.save_memory()

    def snapshot_personality(self, personality_state: dict):
        """Save a snapshot of personality state for evolution tracking."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "creation_count": self.memory["total_creations"],
            "state": personality_state,
        }

        snapshots = self.memory.get("personality_snapshots", [])
        if not isinstance(snapshots, list):
            snapshots = []
        snapshots.append(snapshot)
        self.memory["personality_snapshots"] = snapshots

        # Keep only last 50 snapshots to avoid bloat
        if len(self.memory["personality_snapshots"]) > 50:
            self.memory["personality_snapshots"] = self.memory["personality_snapshots"][
                -50:
            ]

        self.save_memory()

    def get_recent_paintings(self, count: int = 10) -> list[dict]:
        """Get most recent paintings."""
        return self.memory["paintings"][-count:]

    def get_best_paintings(
        self, count: int = 10, min_score: float = 0.65
    ) -> list[dict]:
        """Get highest-scored paintings."""
        scored = [p for p in self.memory["paintings"] if p["score"] >= min_score]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:count]

    def get_paintings_by_mood(self, mood: str) -> list[dict]:
        """Get all paintings created in a specific mood."""
        return [p for p in self.memory["paintings"] if p["mood"] == mood]

    def get_paintings_by_subject(self, subject: str) -> list[dict]:
        """Get all paintings of a specific subject."""
        return [p for p in self.memory["paintings"] if p["subject"] == subject]

    def get_style_statistics(self) -> dict[str, int]:
        """Get count of paintings by style."""
        stats = {}
        for painting in self.memory["paintings"]:
            style = painting.get("style", "unknown")
            stats[style] = stats.get(style, 0) + 1
        return stats

    def get_average_score_by_mood(self) -> dict[str, float]:
        """Calculate average score for each mood."""
        mood_scores = {}
        mood_counts = {}

        for painting in self.memory["paintings"]:
            mood = painting["mood"]
            score = painting["score"]

            mood_scores[mood] = mood_scores.get(mood, 0) + score
            mood_counts[mood] = mood_counts.get(mood, 0) + 1

        return {mood: mood_scores[mood] / mood_counts[mood] for mood in mood_scores}

    def journal_entry(self) -> str:
        """Generate a journal entry summarizing recent creative activity."""
        recent = self.get_recent_paintings(5)
        if not recent:
            return "I haven't created anything yet. The canvas awaits..."

        avg_score = sum(p["score"] for p in recent) / len(recent)
        moods = [p["mood"] for p in recent]
        most_common_mood = max(set(moods), key=moods.count)

        entry = f"I've created {len(recent)} pieces recently. "
        entry += f"I've been feeling mostly {most_common_mood}, "
        entry += f"and my work has been scoring an average of {avg_score:.2f}. "

        if avg_score >= 0.7:
            entry += "I'm quite pleased with my recent output."
        elif avg_score >= 0.5:
            entry += "There's room for growth, but I'm learning."
        else:
            entry += (
                "I'm in an experimental phase - not everything lands, but that's okay."
            )

        return entry

    def import_legacy_memory(self, legacy_file: Path):
        """Import memory from the original autonomous-artist project."""
        if not legacy_file.exists():
            logger.warning("legacy_memory_not_found", file=str(legacy_file))
            return

        try:
            with open(legacy_file) as f:
                legacy = json.load(f)

            # Add legacy paintings as historical
            if "paintings" in legacy:
                for painting in legacy["paintings"]:
                    painting["legacy"] = True
                    painting["imported_at"] = datetime.now().isoformat()
                    self.memory["paintings"].append(painting)

            # Record milestone
            self.record_milestone(
                "Imported legacy memories from previous artistic life",
                {
                    "legacy_file": str(legacy_file),
                    "paintings_imported": len(legacy.get("paintings", [])),
                },
            )

            logger.info(
                "legacy_memory_imported",
                paintings=len(legacy.get("paintings", [])),
                source=str(legacy_file),
            )

            self.save_memory()
        except Exception as e:
            logger.error("legacy_import_failed", error=str(e))
