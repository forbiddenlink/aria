"""Tests for Aria's mood system."""

import random
from unittest.mock import patch

import pytest

from ai_artist.personality.moods import Mood, MoodSystem


class TestMood:
    """Test the Mood enum."""

    def test_mood_values(self):
        """Test that all expected moods exist."""
        expected = [
            "contemplative",
            "chaotic",
            "melancholic",
            "energized",
            "rebellious",
            "serene",
            "restless",
            "playful",
            "introspective",
            "bold",
        ]
        actual = [m.value for m in Mood]
        assert sorted(actual) == sorted(expected)

    def test_mood_is_string_enum(self):
        """Test that Mood is a string enum."""
        assert isinstance(Mood.CONTEMPLATIVE.value, str)
        assert str(Mood.CONTEMPLATIVE) == "Mood.CONTEMPLATIVE"
        assert Mood.CONTEMPLATIVE == "contemplative"


class TestMoodSystem:
    """Test the MoodSystem class."""

    @pytest.fixture
    def mood_system(self):
        """Create a fresh MoodSystem for each test."""
        return MoodSystem()

    def test_initialization(self, mood_system):
        """Test MoodSystem initializes with correct defaults."""
        assert mood_system.current_mood == Mood.CONTEMPLATIVE
        assert mood_system.energy_level == 0.5
        assert mood_system.mood_duration == 0
        assert isinstance(mood_system.mood_influences, dict)
        assert len(mood_system.mood_influences) == len(Mood)

    def test_initial_mood_is_contemplative(self, mood_system):
        """Test that Aria starts contemplative."""
        assert mood_system.current_mood == Mood.CONTEMPLATIVE

    def test_energy_level_bounds(self, mood_system):
        """Test that energy level stays within 0-1 bounds."""
        # Force energy to extremes through multiple updates
        for _ in range(100):
            mood_system.update_mood()
            assert 0.0 <= mood_system.energy_level <= 1.0

    def test_mood_duration_increments(self, mood_system):
        """Test that mood duration increments on update."""
        initial_duration = mood_system.mood_duration
        mood_system.update_mood()
        assert mood_system.mood_duration > initial_duration

    def test_mood_influences_have_all_keys(self, mood_system):
        """Test that every mood has styles, colors, and subjects."""
        for mood in Mood:
            influences = mood_system.mood_influences[mood]
            assert "styles" in influences
            assert "colors" in influences
            assert "subjects" in influences
            assert isinstance(influences["styles"], list)
            assert isinstance(influences["colors"], list)
            assert isinstance(influences["subjects"], list)
            assert len(influences["styles"]) > 0
            assert len(influences["colors"]) > 0
            assert len(influences["subjects"]) > 0

    def test_update_mood_returns_mood(self, mood_system):
        """Test update_mood returns a Mood enum."""
        result = mood_system.update_mood()
        assert isinstance(result, Mood)

    def test_mood_shift_after_duration(self, mood_system):
        """Test that mood shifts after sufficient duration."""
        # Force a short duration threshold
        with patch.object(random, "randint", return_value=1):
            # First update increments duration to 1 (triggers shift since 1 > 1 is false)
            mood_system.update_mood()
            # Check duration is now 1
            assert mood_system.mood_duration == 1
            # Second update: duration becomes 2, which is > 1, triggers shift
            mood_system.update_mood()
            # After shift, duration is reset to 0 then incremented to 1
            # But actually the shift happens during update, so duration is reset to 0
            # The increment happens before the shift check in update_mood
            # Looking at the code: duration increments first, then checks if > threshold
            # So: start 0 -> increment to 1 -> 1 > 1? No -> no shift
            # Then: start 1 -> increment to 2 -> 2 > 1? Yes -> shift resets to 0
            assert mood_system.mood_duration == 0  # Reset after shift

    def test_mood_transitions_are_valid(self, mood_system):
        """Test that mood transitions follow defined pathways."""
        # Test each mood's possible transitions
        transitions = {
            Mood.CONTEMPLATIVE: [Mood.SERENE, Mood.INTROSPECTIVE, Mood.MELANCHOLIC],
            Mood.CHAOTIC: [Mood.REBELLIOUS, Mood.ENERGIZED, Mood.RESTLESS],
            Mood.MELANCHOLIC: [Mood.CONTEMPLATIVE, Mood.INTROSPECTIVE, Mood.SERENE],
            Mood.ENERGIZED: [Mood.PLAYFUL, Mood.BOLD, Mood.CHAOTIC],
            Mood.REBELLIOUS: [Mood.CHAOTIC, Mood.BOLD, Mood.RESTLESS],
            Mood.SERENE: [Mood.CONTEMPLATIVE, Mood.PLAYFUL, Mood.INTROSPECTIVE],
            Mood.RESTLESS: [Mood.CHAOTIC, Mood.REBELLIOUS, Mood.INTROSPECTIVE],
            Mood.PLAYFUL: [Mood.ENERGIZED, Mood.SERENE, Mood.BOLD],
            Mood.INTROSPECTIVE: [Mood.CONTEMPLATIVE, Mood.MELANCHOLIC, Mood.SERENE],
            Mood.BOLD: [Mood.REBELLIOUS, Mood.ENERGIZED, Mood.PLAYFUL],
        }

        for start_mood, valid_transitions in transitions.items():
            mood_system.current_mood = start_mood
            mood_system._shift_mood()
            assert mood_system.current_mood in valid_transitions

    def test_contemplative_to_serene_transition(self, mood_system):
        """Test specific transition: contemplative can become serene."""
        mood_system.current_mood = Mood.CONTEMPLATIVE
        # Run multiple transitions to find one
        found_serene = False
        for _ in range(100):
            mood_system.current_mood = Mood.CONTEMPLATIVE
            mood_system._shift_mood()
            if mood_system.current_mood == Mood.SERENE:
                found_serene = True
                break
        assert found_serene, "Serene should be reachable from contemplative"


class TestMoodInfluenceOnPrompts:
    """Test how moods influence prompts."""

    @pytest.fixture
    def mood_system(self):
        return MoodSystem()

    def test_influence_prompt_adds_content(self, mood_system):
        """Test that influence_prompt modifies the base prompt."""
        base_prompt = "a simple landscape"
        influenced = mood_system.influence_prompt(base_prompt)
        assert len(influenced) > len(base_prompt)
        assert base_prompt in influenced

    def test_influence_prompt_adds_mood_descriptor(self, mood_system):
        """Test that mood descriptor is always added."""
        base_prompt = "test image"
        influenced = mood_system.influence_prompt(base_prompt)
        # Should contain mood descriptor
        assert "mood" in influenced.lower()

    def test_contemplative_mood_adds_quiet_thoughtful(self, mood_system):
        """Test contemplative mood adds its descriptor."""
        mood_system.current_mood = Mood.CONTEMPLATIVE
        influenced = mood_system.influence_prompt("test")
        assert "quiet" in influenced or "thoughtful" in influenced

    def test_chaotic_mood_adds_wild_energetic(self, mood_system):
        """Test chaotic mood adds its descriptor."""
        mood_system.current_mood = Mood.CHAOTIC
        influenced = mood_system.influence_prompt("test")
        assert "wild" in influenced or "energetic" in influenced

    def test_all_moods_have_descriptors(self, mood_system):
        """Test every mood adds some descriptor."""
        base = "test prompt"
        for mood in Mood:
            mood_system.current_mood = mood
            influenced = mood_system.influence_prompt(base)
            assert influenced != base
            assert "mood" in influenced


class TestMoodDurationTracking:
    """Test mood duration tracking."""

    @pytest.fixture
    def mood_system(self):
        return MoodSystem()

    def test_duration_starts_at_zero(self, mood_system):
        """Test duration starts at zero."""
        assert mood_system.mood_duration == 0

    def test_duration_increments_on_update(self, mood_system):
        """Test duration increments with each update."""
        for i in range(5):
            mood_system.mood_duration = i  # Reset to known value
            with patch.object(random, "randint", return_value=100):  # Never shift
                mood_system.update_mood()
            assert mood_system.mood_duration == i + 1

    def test_duration_resets_on_shift(self, mood_system):
        """Test duration resets to 0 after mood shift."""
        mood_system.mood_duration = 10
        mood_system._shift_mood()
        assert mood_system.mood_duration == 0


class TestExternalFactorsAffectingMood:
    """Test how external factors affect mood (placeholder for future)."""

    @pytest.fixture
    def mood_system(self):
        return MoodSystem()

    def test_update_mood_accepts_external_factors(self, mood_system):
        """Test that update_mood accepts external factors parameter."""
        # Currently external_factors is not used, but the parameter exists
        result = mood_system.update_mood(external_factors={"weather": "sunny"})
        assert isinstance(result, Mood)

    def test_external_factors_parameter_is_optional(self, mood_system):
        """Test that external_factors is optional."""
        result = mood_system.update_mood()
        assert isinstance(result, Mood)


class TestMoodBasedSubjects:
    """Test mood-based subject selection."""

    @pytest.fixture
    def mood_system(self):
        return MoodSystem()

    def test_get_mood_based_subject_returns_string(self, mood_system):
        """Test get_mood_based_subject returns a string."""
        subject = mood_system.get_mood_based_subject()
        assert isinstance(subject, str)
        assert len(subject) > 0

    def test_subject_matches_mood(self, mood_system):
        """Test subjects come from current mood's pool."""
        for mood in Mood:
            mood_system.current_mood = mood
            subject = mood_system.get_mood_based_subject()
            expected_subjects = mood_system.mood_influences[mood]["subjects"]
            assert subject in expected_subjects


class TestMoodDescriptions:
    """Test mood description methods."""

    @pytest.fixture
    def mood_system(self):
        return MoodSystem()

    def test_describe_feeling_returns_string(self, mood_system):
        """Test describe_feeling returns a non-empty string."""
        feeling = mood_system.describe_feeling()
        assert isinstance(feeling, str)
        assert len(feeling) > 0

    def test_describe_feeling_includes_energy(self, mood_system):
        """Test describe_feeling includes energy level."""
        feeling = mood_system.describe_feeling()
        assert "Energy" in feeling

    def test_describe_feeling_varies_by_mood(self, mood_system):
        """Test different moods produce different descriptions."""
        feelings = set()
        for mood in Mood:
            mood_system.current_mood = mood
            feeling = mood_system.describe_feeling()
            feelings.add(feeling.split("(")[0].strip())  # Remove energy part
        assert len(feelings) == len(Mood)


class TestMoodReflections:
    """Test mood-based reflections on work."""

    @pytest.fixture
    def mood_system(self):
        return MoodSystem()

    def test_reflect_on_work_returns_string(self, mood_system):
        """Test reflect_on_work returns a non-empty string."""
        reflection = mood_system.reflect_on_work(score=0.7, subject="sunset")
        assert isinstance(reflection, str)
        assert len(reflection) > 0

    def test_reflect_on_work_includes_subject(self, mood_system):
        """Test reflection includes the subject."""
        reflection = mood_system.reflect_on_work(score=0.7, subject="mountains")
        assert "mountains" in reflection

    def test_high_score_adds_positive_sentiment(self, mood_system):
        """Test high scores add positive sentiment."""
        reflection = mood_system.reflect_on_work(score=0.9, subject="test")
        assert "pleased" in reflection or "satisfied" in reflection.lower()

    def test_low_score_adds_growth_sentiment(self, mood_system):
        """Test low scores add growth-oriented sentiment."""
        reflection = mood_system.reflect_on_work(score=0.3, subject="test")
        assert "journey" in reflection or "envision" in reflection

    def test_medium_score_is_neutral(self, mood_system):
        """Test medium scores are neutral."""
        reflection = mood_system.reflect_on_work(score=0.5, subject="test")
        assert "interesting" in reflection or "best" in reflection


class TestMoodColorAndStyle:
    """Test mood-based color and style methods."""

    @pytest.fixture
    def mood_system(self):
        return MoodSystem()

    def test_get_mood_colors_returns_list(self, mood_system):
        """Test get_mood_colors returns a list."""
        colors = mood_system.get_mood_colors()
        assert isinstance(colors, list)
        assert len(colors) > 0

    def test_get_mood_colors_matches_influences(self, mood_system):
        """Test colors match the mood_influences definition."""
        for mood in Mood:
            mood_system.current_mood = mood
            colors = mood_system.get_mood_colors()
            expected = mood_system.mood_influences[mood]["colors"]
            assert colors == expected

    def test_get_mood_style_returns_string(self, mood_system):
        """Test get_mood_style returns a string."""
        style = mood_system.get_mood_style()
        assert isinstance(style, str)
        assert len(style) > 0

    def test_get_mood_style_from_influences(self, mood_system):
        """Test style comes from mood influences."""
        for mood in Mood:
            mood_system.current_mood = mood
            style = mood_system.get_mood_style()
            expected_styles = mood_system.mood_influences[mood]["styles"]
            assert style in expected_styles
