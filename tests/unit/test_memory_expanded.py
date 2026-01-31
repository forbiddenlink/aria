"""Expanded tests for Aria's memory systems.

Tests both the basic ArtistMemory and the EnhancedMemorySystem.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ai_artist.personality.memory import ArtistMemory
from ai_artist.personality.enhanced_memory import (
    EnhancedMemorySystem,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
)


# ============================================================================
# Tests for ArtistMemory (basic memory system)
# ============================================================================


class TestArtistMemoryInit:
    """Test ArtistMemory initialization."""

    def test_init_with_default_path(self, tmp_path):
        """Test initialization with default path."""
        with patch.object(ArtistMemory, "_load"):
            memory = ArtistMemory()
            assert memory.memory_file == Path("data/aria_memory.json")

    def test_init_with_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        custom_path = tmp_path / "custom_memory.json"
        with patch.object(ArtistMemory, "_load"):
            memory = ArtistMemory(memory_file=custom_path)
            assert memory.memory_file == custom_path

    def test_init_creates_default_structure(self, tmp_path):
        """Test initialization creates default memory structure."""
        memory_path = tmp_path / "memory.json"
        memory = ArtistMemory(memory_file=memory_path)

        assert memory.memory["name"] == "Aria"
        assert "paintings" in memory.memory
        assert "reflections" in memory.memory
        assert "preferences" in memory.memory
        assert "stats" in memory.memory


class TestEpisodicMemoryRecording:
    """Test episodic memory recording."""

    @pytest.fixture
    def memory(self, tmp_path):
        """Create an ArtistMemory instance."""
        return ArtistMemory(memory_file=tmp_path / "test_memory.json")

    def test_remember_artwork_adds_entry(self, memory):
        """Test remember_artwork adds a new entry."""
        memory.remember_artwork(
            prompt="a sunset over ocean",
            subject="sunset",
            style="impressionist",
            mood="serene",
            colors=["orange", "pink", "blue"],
            score=0.85,
            image_path="/path/to/image.png",
        )

        assert len(memory.memory["paintings"]) == 1
        painting = memory.memory["paintings"][0]
        assert painting["subject"] == "sunset"
        assert painting["score"] == 0.85

    def test_remember_artwork_updates_total(self, memory):
        """Test remember_artwork increments total count."""
        assert memory.memory["stats"]["total_created"] == 0

        memory.remember_artwork(
            prompt="test",
            subject="test",
            style="test",
            mood="serene",
            colors=[],
            score=0.5,
            image_path="/test.png",
        )

        assert memory.memory["stats"]["total_created"] == 1

    def test_remember_artwork_updates_best_score(self, memory):
        """Test remember_artwork updates best score."""
        memory.remember_artwork(
            prompt="test1",
            subject="a",
            style="x",
            mood="serene",
            colors=[],
            score=0.6,
            image_path="/a.png",
        )
        assert memory.memory["stats"]["best_score"] == 0.6

        memory.remember_artwork(
            prompt="test2",
            subject="b",
            style="y",
            mood="chaotic",
            colors=[],
            score=0.9,
            image_path="/b.png",
        )
        assert memory.memory["stats"]["best_score"] == 0.9

        # Lower score shouldn't replace
        memory.remember_artwork(
            prompt="test3",
            subject="c",
            style="z",
            mood="bold",
            colors=[],
            score=0.7,
            image_path="/c.png",
        )
        assert memory.memory["stats"]["best_score"] == 0.9


class TestSemanticLearning:
    """Test semantic learning through preferences."""

    @pytest.fixture
    def memory(self, tmp_path):
        return ArtistMemory(memory_file=tmp_path / "test.json")

    def test_update_preferences_tracks_subjects(self, memory):
        """Test preferences track subject frequency."""
        memory.remember_artwork("p", "ocean", "s", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "ocean", "s", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "mountain", "s", "m", [], 0.5, "/i.png")

        subjects = memory.memory["preferences"]["favorite_subjects"]
        assert subjects["ocean"] == 2
        assert subjects["mountain"] == 1

    def test_update_preferences_tracks_styles(self, memory):
        """Test preferences track style frequency."""
        memory.remember_artwork("p", "s", "impressionist", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "s", "impressionist", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "s", "minimalist", "m", [], 0.5, "/i.png")

        styles = memory.memory["preferences"]["favorite_styles"]
        assert styles["impressionist"] == 2
        assert styles["minimalist"] == 1

    def test_update_preferences_tracks_colors(self, memory):
        """Test preferences track color frequency."""
        memory.remember_artwork("p", "s", "st", "m", ["blue", "green"], 0.5, "/i.png")
        memory.remember_artwork("p", "s", "st", "m", ["blue", "red"], 0.5, "/i.png")

        colors = memory.memory["preferences"]["favorite_colors"]
        assert colors["blue"] == 2
        assert colors["green"] == 1
        assert colors["red"] == 1


class TestRelevanceSearch:
    """Test relevance search in memory."""

    @pytest.fixture
    def memory(self, tmp_path):
        return ArtistMemory(memory_file=tmp_path / "test.json")

    def test_get_recent_works(self, memory):
        """Test getting recent works."""
        for i in range(15):
            memory.remember_artwork(f"p{i}", f"s{i}", "st", "m", [], 0.5, f"/{i}.png")

        recent = memory.get_recent_works(limit=10)
        assert len(recent) == 10
        # Should be the last 10
        assert recent[0]["subject"] == "s5"
        assert recent[-1]["subject"] == "s14"

    def test_get_best_works(self, memory):
        """Test getting best scored works."""
        scores = [0.3, 0.9, 0.5, 0.8, 0.2]
        for i, score in enumerate(scores):
            memory.remember_artwork(f"p{i}", f"s{i}", "st", "m", [], score, f"/{i}.png")

        best = memory.get_best_works(limit=3)
        assert len(best) == 3
        assert best[0]["score"] == 0.9
        assert best[1]["score"] == 0.8
        assert best[2]["score"] == 0.5

    def test_has_painted_recently_true(self, memory):
        """Test has_painted_recently returns True for recent subject."""
        memory.remember_artwork("p", "ocean", "st", "m", [], 0.5, "/i.png")
        assert memory.has_painted_recently("ocean") is True

    def test_has_painted_recently_false(self, memory):
        """Test has_painted_recently returns False for old subject."""
        # Add old painting
        memory.remember_artwork("p", "ocean", "st", "m", [], 0.5, "/i.png")
        # Add 5 more to push it beyond threshold
        for i in range(5):
            memory.remember_artwork("p", f"other{i}", "st", "m", [], 0.5, f"/{i}.png")

        assert memory.has_painted_recently("ocean", threshold=5) is False

    def test_get_favorite_subject(self, memory):
        """Test get_favorite_subject returns most frequent."""
        memory.remember_artwork("p", "a", "st", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "b", "st", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "b", "st", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "b", "st", "m", [], 0.5, "/i.png")
        memory.remember_artwork("p", "a", "st", "m", [], 0.5, "/i.png")

        assert memory.get_favorite_subject() == "b"

    def test_get_favorite_subject_empty(self, memory):
        """Test get_favorite_subject returns None when empty."""
        assert memory.get_favorite_subject() is None


class TestMemoryConsolidation:
    """Test memory persistence (consolidation)."""

    def test_save_creates_file(self, tmp_path):
        """Test save creates file on disk."""
        memory_path = tmp_path / "memory.json"
        memory = ArtistMemory(memory_file=memory_path)
        memory.remember_artwork("p", "s", "st", "m", [], 0.5, "/i.png")

        assert memory_path.exists()

    def test_load_restores_memory(self, tmp_path):
        """Test load restores saved memory."""
        memory_path = tmp_path / "memory.json"

        # Create and save
        memory1 = ArtistMemory(memory_file=memory_path)
        memory1.remember_artwork("p", "test_subject", "st", "m", [], 0.8, "/i.png")

        # Load in new instance
        memory2 = ArtistMemory(memory_file=memory_path)

        assert len(memory2.memory["paintings"]) == 1
        assert memory2.memory["paintings"][0]["subject"] == "test_subject"


# ============================================================================
# Tests for EpisodicMemory (enhanced memory)
# ============================================================================


class TestEpisodicMemory:
    """Test the EpisodicMemory class."""

    @pytest.fixture
    def episodic(self):
        return EpisodicMemory()

    def test_record_episode(self, episodic):
        """Test recording an episode."""
        episodic.record_episode(
            event_type="creation",
            details={"subject": "sunset"},
            emotional_state={"mood": "serene", "energy": 0.7},
        )

        assert len(episodic.episodes) == 1
        ep = episodic.episodes[0]
        assert ep["event_type"] == "creation"
        assert ep["details"]["subject"] == "sunset"

    def test_record_episode_with_timestamp(self, episodic):
        """Test recording episode with custom timestamp."""
        episodic.record_episode(
            event_type="test",
            details={},
            emotional_state={},
            timestamp="2024-01-01T12:00:00",
        )

        assert episodic.episodes[0]["timestamp"] == "2024-01-01T12:00:00"

    def test_get_recent_episodes(self, episodic):
        """Test getting recent episodes."""
        for i in range(10):
            episodic.record_episode(f"type{i}", {"i": i}, {})

        recent = episodic.get_recent_episodes(count=5)
        assert len(recent) == 5
        assert recent[0]["details"]["i"] == 5
        assert recent[-1]["details"]["i"] == 9

    def test_get_recent_episodes_filtered(self, episodic):
        """Test getting recent episodes filtered by type."""
        episodic.record_episode("creation", {"i": 1}, {})
        episodic.record_episode("reflection", {"i": 2}, {})
        episodic.record_episode("creation", {"i": 3}, {})

        creations = episodic.get_recent_episodes(count=10, event_type="creation")
        assert len(creations) == 2
        assert all(ep["event_type"] == "creation" for ep in creations)

    def test_get_episodes_by_mood(self, episodic):
        """Test getting episodes by mood."""
        episodic.record_episode("a", {}, {"mood": "serene"})
        episodic.record_episode("b", {}, {"mood": "chaotic"})
        episodic.record_episode("c", {}, {"mood": "serene"})

        serene = episodic.get_episodes_by_mood("serene")
        assert len(serene) == 2

    def test_count_episodes_by_type(self, episodic):
        """Test counting episodes by type."""
        episodic.record_episode("creation", {}, {})
        episodic.record_episode("creation", {}, {})
        episodic.record_episode("reflection", {}, {})

        counts = episodic.count_episodes_by_type()
        assert counts["creation"] == 2
        assert counts["reflection"] == 1


# ============================================================================
# Tests for SemanticMemory (enhanced memory)
# ============================================================================


class TestSemanticMemory:
    """Test the SemanticMemory class."""

    @pytest.fixture
    def semantic(self):
        return SemanticMemory()

    def test_record_style_effectiveness(self, semantic):
        """Test recording style effectiveness."""
        semantic.record_style_effectiveness("impressionist", 0.85, 10)

        assert "impressionist" in semantic.knowledge["style_effectiveness"]
        assert semantic.knowledge["style_effectiveness"]["impressionist"]["avg_score"] == 0.85
        assert semantic.knowledge["style_effectiveness"]["impressionist"]["sample_size"] == 10

    def test_record_subject_resonance(self, semantic):
        """Test recording subject resonance."""
        semantic.record_subject_resonance("ocean", {"beauty": 0.9, "depth": 0.8})

        assert "ocean" in semantic.knowledge["subject_resonance"]
        assert semantic.knowledge["subject_resonance"]["ocean"]["beauty"] == 0.9

    def test_learn_association(self, semantic):
        """Test learning an association."""
        semantic.learn_association("Blue and orange create tension", "color_theory")

        associations = semantic.knowledge["learned_associations"]
        assert len(associations) == 1
        assert associations[0]["insight"] == "Blue and orange create tension"
        assert associations[0]["category"] == "color_theory"

    def test_get_best_styles(self, semantic):
        """Test getting best styles."""
        semantic.record_style_effectiveness("style_a", 0.7, 5)
        semantic.record_style_effectiveness("style_b", 0.9, 10)
        semantic.record_style_effectiveness("style_c", 0.8, 3)  # Too few samples
        semantic.record_style_effectiveness("style_d", 0.6, 8)

        best = semantic.get_best_styles(min_samples=5)
        assert len(best) == 3
        assert best[0][0] == "style_b"  # Highest score
        assert best[0][1] == 0.9

    def test_get_insights_by_category(self, semantic):
        """Test getting insights by category."""
        semantic.learn_association("Insight 1", "color")
        semantic.learn_association("Insight 2", "composition")
        semantic.learn_association("Insight 3", "color")

        color_insights = semantic.get_insights_by_category("color")
        assert len(color_insights) == 2
        assert "Insight 1" in color_insights
        assert "Insight 3" in color_insights


# ============================================================================
# Tests for WorkingMemory (enhanced memory)
# ============================================================================


class TestWorkingMemory:
    """Test the WorkingMemory class."""

    @pytest.fixture
    def working(self):
        return WorkingMemory()

    def test_set_and_get_context(self, working):
        """Test setting and getting context."""
        working.set_context("current_subject", "sunset")
        assert working.get_context("current_subject") == "sunset"

    def test_get_context_missing_key(self, working):
        """Test getting missing context returns None."""
        assert working.get_context("nonexistent") is None

    def test_add_goal(self, working):
        """Test adding a goal."""
        working.add_goal("Create a painting")
        assert "Create a painting" in working.active_goals

    def test_complete_goal(self, working):
        """Test completing a goal."""
        working.add_goal("Goal 1")
        working.add_goal("Goal 2")
        working.complete_goal("Goal 1")

        assert "Goal 1" not in working.active_goals
        assert "Goal 2" in working.active_goals

    def test_complete_nonexistent_goal(self, working):
        """Test completing nonexistent goal doesn't raise."""
        working.complete_goal("Nonexistent")  # Should not raise

    def test_clear_session(self, working):
        """Test clearing session."""
        working.set_context("key", "value")
        working.add_goal("goal")
        old_start = working.session_start

        import time
        time.sleep(0.01)  # Ensure different timestamp
        working.clear_session()

        assert working.current_context == {}
        assert working.active_goals == []
        assert working.session_start != old_start


# ============================================================================
# Tests for EnhancedMemorySystem
# ============================================================================


class TestEnhancedMemorySystem:
    """Test the integrated EnhancedMemorySystem."""

    @pytest.fixture
    def memory_system(self, tmp_path):
        return EnhancedMemorySystem(memory_file=tmp_path / "enhanced.json")

    def test_initialization(self, memory_system):
        """Test system initializes all memory types."""
        assert isinstance(memory_system.episodic, EpisodicMemory)
        assert isinstance(memory_system.semantic, SemanticMemory)
        assert isinstance(memory_system.working, WorkingMemory)

    def test_record_creation_updates_both_memories(self, memory_system):
        """Test record_creation updates episodic and semantic."""
        memory_system.record_creation(
            artwork_details={
                "prompt": "test prompt",
                "style": "impressionist",
                "subject": "ocean",
            },
            emotional_state={
                "mood": "serene",
                "energy_level": 0.7,
            },
            outcome={
                "score": 0.85,
            },
        )

        # Check episodic
        assert len(memory_system.episodic.episodes) == 1

        # Check semantic (style effectiveness updated)
        assert "impressionist" in memory_system.semantic.knowledge["style_effectiveness"]

    def test_record_creation_calculates_running_average(self, memory_system):
        """Test style effectiveness uses running average."""
        memory_system.record_creation(
            artwork_details={"style": "test_style"},
            emotional_state={},
            outcome={"score": 0.8},
        )
        memory_system.record_creation(
            artwork_details={"style": "test_style"},
            emotional_state={},
            outcome={"score": 0.6},
        )

        effectiveness = memory_system.semantic.knowledge["style_effectiveness"]["test_style"]
        assert effectiveness["sample_size"] == 2
        assert effectiveness["avg_score"] == 0.7  # (0.8 + 0.6) / 2

    def test_generate_insights(self, memory_system):
        """Test generate_insights produces meaningful output."""
        memory_system.record_creation(
            artwork_details={"style": "impressionist"},
            emotional_state={"mood": "serene"},
            outcome={"score": 0.85},
        )
        memory_system.record_creation(
            artwork_details={"style": "impressionist"},
            emotional_state={"mood": "chaotic"},
            outcome={"score": 0.75},
        )

        insights = memory_system.generate_insights()

        assert insights["total_creations"] == 2
        assert "impressionist" in insights["style_effectiveness"]
        assert "serene" in insights["mood_patterns"]

    def test_get_relevant_context(self, memory_system):
        """Test get_relevant_context retrieves appropriate memories."""
        memory_system.episodic.record_episode(
            "creation",
            {"subject": "ocean"},
            {"mood": "serene"},
        )
        memory_system.semantic.record_style_effectiveness("minimalist", 0.9, 5)

        context = memory_system.get_relevant_context("serene", limit=5)

        assert "similar_mood_episodes" in context
        assert "best_styles" in context
        assert "recent_insights" in context

    def test_save_and_load(self, tmp_path):
        """Test persistence across instances."""
        memory_path = tmp_path / "persist.json"

        # Create and populate
        memory1 = EnhancedMemorySystem(memory_file=memory_path)
        memory1.record_creation(
            artwork_details={"style": "test", "subject": "test_subject"},
            emotional_state={"mood": "contemplative"},
            outcome={"score": 0.9},
        )

        # Load in new instance
        memory2 = EnhancedMemorySystem(memory_file=memory_path)

        assert len(memory2.episodic.episodes) == 1
        assert "test" in memory2.semantic.knowledge["style_effectiveness"]

    def test_working_memory_not_persisted(self, tmp_path):
        """Test working memory is not persisted."""
        memory_path = tmp_path / "working_test.json"

        memory1 = EnhancedMemorySystem(memory_file=memory_path)
        memory1.working.set_context("session_data", "temporary")
        memory1.save()

        memory2 = EnhancedMemorySystem(memory_file=memory_path)
        assert memory2.working.get_context("session_data") is None


class TestMemoryReflectionGeneration:
    """Test memory-based reflection generation."""

    @pytest.fixture
    def memory(self, tmp_path):
        return ArtistMemory(memory_file=tmp_path / "test.json")

    def test_generate_reflection_includes_subject(self, memory):
        """Test generated reflection includes subject."""
        artwork = {
            "subject": "mountains",
            "style": "impressionist",
            "mood": "serene",
            "score": 0.8,
        }
        reflection = memory.generate_reflection(artwork)
        assert "mountains" in reflection

    def test_generate_reflection_includes_artwork_details(self, memory):
        """Test generated reflection includes some artwork details."""
        artwork = {
            "subject": "sunset",
            "style": "impressionist",
            "mood": "contemplative",
            "score": 0.5,
        }
        # Run multiple times since reflections are randomly selected
        found_subject = False
        for _ in range(20):
            reflection = memory.generate_reflection(artwork)
            if "sunset" in reflection:
                found_subject = True
                break
        assert found_subject, "At least one reflection should mention the subject"

    def test_generate_reflection_varies(self, memory):
        """Test reflections have variety."""
        artwork = {
            "subject": "test",
            "style": "test",
            "mood": "serene",
            "score": 0.7,
        }
        reflections = set()
        for _ in range(20):
            reflections.add(memory.generate_reflection(artwork))

        # Should have some variety
        assert len(reflections) > 1


class TestMemoryStats:
    """Test memory statistics."""

    @pytest.fixture
    def memory(self, tmp_path):
        return ArtistMemory(memory_file=tmp_path / "test.json")

    def test_get_stats_empty(self, memory):
        """Test stats with empty memory."""
        stats = memory.get_stats()
        assert stats["total_artworks"] == 0
        assert stats["best_score"] == 0.0
        assert stats["favorite_subject"] is None

    def test_get_stats_populated(self, memory):
        """Test stats with populated memory."""
        memory.remember_artwork("p", "ocean", "impressionist", "m", [], 0.8, "/i.png")
        memory.remember_artwork("p", "ocean", "minimalist", "m", [], 0.9, "/i.png")
        memory.remember_artwork("p", "mountain", "impressionist", "m", [], 0.7, "/i.png")

        stats = memory.get_stats()

        assert stats["total_artworks"] == 3
        assert stats["best_score"] == 0.9
        assert stats["favorite_subject"] == "ocean"
        assert stats["favorite_style"] == "impressionist"
