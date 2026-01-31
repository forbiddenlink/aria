"""
Tests for new prompt utilities: prompt matrix, emphasis, and style presets.
"""

import tempfile
from pathlib import Path

import pytest

from ai_artist.utils.prompt_emphasis import PromptEmphasis
from ai_artist.utils.prompt_matrix import PromptMatrix
from ai_artist.utils.style_presets import StylePreset, StylePresetsManager


class TestPromptMatrix:
    """Test the prompt matrix generator."""

    def test_simple_matrix(self):
        """Test basic matrix generation."""
        pm = PromptMatrix()
        result = pm.parse_prompt("a {red|blue} cat")

        assert len(result) == 2
        assert "a red cat" in result
        assert "a blue cat" in result

    def test_multiple_matrices(self):
        """Test multiple matrix groups."""
        pm = PromptMatrix()
        result = pm.parse_prompt("{big|small} {red|blue} cat")

        assert len(result) == 4
        assert "big red cat" in result
        assert "big blue cat" in result
        assert "small red cat" in result
        assert "small blue cat" in result

    def test_no_matrix(self):
        """Test prompt without matrix syntax."""
        pm = PromptMatrix()
        result = pm.parse_prompt("simple prompt")

        assert len(result) == 1
        assert result[0] == "simple prompt"

    def test_count_combinations(self):
        """Test counting combinations."""
        pm = PromptMatrix()

        assert pm.count_combinations("no matrix") == 1
        assert pm.count_combinations("{a|b} test") == 2
        assert pm.count_combinations("{a|b|c} {x|y}") == 6

    def test_validation(self):
        """Test syntax validation."""
        pm = PromptMatrix()

        # Valid
        is_valid, _ = pm.validate_syntax("{a|b} test")
        assert is_valid

        # Unmatched braces
        is_valid, error = pm.validate_syntax("{a|b test")
        assert not is_valid
        assert "Unmatched" in error

        # Empty option
        is_valid, error = pm.validate_syntax("{a||b}")
        assert not is_valid
        assert "Empty" in error


class TestPromptEmphasis:
    """Test the prompt emphasis system."""

    def test_parse_simple_emphasis(self):
        """Test parsing basic emphasis."""
        pe = PromptEmphasis()
        result = pe.parse_emphasis("(beautiful:1.5) woman")

        assert len(result) == 2
        assert result[0] == ("beautiful", 1.5)
        assert result[1] == ("woman", 1.0)

    def test_default_emphasis(self):
        """Test default emphasis without weight."""
        pe = PromptEmphasis()
        result = pe.parse_emphasis("(beautiful) woman")

        assert result[0][1] == 1.1  # Default emphasis

    def test_compel_conversion(self):
        """Test conversion to Compel format."""
        pe = PromptEmphasis()

        # Strong emphasis
        result = pe.apply_emphasis_to_compel("(beautiful:1.5) woman")
        assert "+" in result

        # De-emphasis
        result = pe.apply_emphasis_to_compel("(background:0.5) details")
        assert "-" in result

    def test_no_emphasis(self):
        """Test prompt without emphasis."""
        pe = PromptEmphasis()
        result = pe.parse_emphasis("simple prompt")

        assert len(result) == 1
        assert result[0] == ("simple prompt", 1.0)

    def test_validation(self):
        """Test emphasis syntax validation."""
        pe = PromptEmphasis()

        # Valid
        is_valid, _ = pe.validate_syntax("(text:1.5) prompt")
        assert is_valid

        # Unmatched parentheses
        is_valid, error = pe.validate_syntax("(text:1.5 prompt")
        assert not is_valid
        assert "Unmatched" in error

    def test_average_weight(self):
        """Test average weight calculation."""
        pe = PromptEmphasis()

        avg = pe.get_effective_weight("(a:1.5) (b:0.5)")
        assert avg == 1.0  # (1.5 + 0.5) / 2


class TestStylePresets:
    """Test the style presets system."""

    def test_preset_creation(self):
        """Test creating a style preset."""
        preset = StylePreset(
            name="Test",
            positive="test positive",
            negative="test negative",
            description="test preset",
            category="test",
        )

        assert preset.name == "Test"
        assert preset.positive == "test positive"
        assert preset.negative == "test negative"

    def test_apply_without_placeholder(self):
        """Test applying preset without {prompt} placeholder."""
        preset = StylePreset(
            name="Test", positive="cinematic, dramatic", negative="amateur"
        )

        result = preset.apply_to_prompt("portrait")
        assert result == "portrait, cinematic, dramatic"

    def test_apply_with_placeholder(self):
        """Test applying preset with {prompt} placeholder."""
        preset = StylePreset(
            name="Test",
            positive="oil painting of {prompt}, masterpiece",
            negative="digital",
        )

        result = preset.apply_to_prompt("a cat")
        assert result == "oil painting of a cat, masterpiece"

    def test_manager_initialization(self):
        """Test preset manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            presets_file = Path(tmpdir) / "presets.json"
            manager = StylePresetsManager(presets_file)

            # Should have default presets
            presets = manager.list_presets()
            assert len(presets) > 0
            assert any(p.name == "Cinematic" for p in presets)

    def test_save_and_load(self):
        """Test saving and loading presets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            presets_file = Path(tmpdir) / "presets.json"

            # Create and save
            manager1 = StylePresetsManager(presets_file)
            original_count = len(manager1.presets)

            # Add custom preset
            custom = StylePreset(
                name="Custom", positive="custom style", negative="unwanted"
            )
            manager1.add_preset(custom)

            # Load in new manager
            manager2 = StylePresetsManager(presets_file)
            assert len(manager2.presets) == original_count + 1
            assert manager2.get_preset("Custom") is not None

    def test_list_by_category(self):
        """Test filtering presets by category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            presets_file = Path(tmpdir) / "presets.json"
            manager = StylePresetsManager(presets_file)

            art_presets = manager.list_presets(category="art")
            assert len(art_presets) > 0
            assert all(p.category == "art" for p in art_presets)

    def test_apply_preset(self):
        """Test applying a preset through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            presets_file = Path(tmpdir) / "presets.json"
            manager = StylePresetsManager(presets_file)

            positive, negative = manager.apply_preset("Cinematic", "portrait")

            assert "portrait" in positive
            assert "cinematic" in positive.lower()
            assert len(negative) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
