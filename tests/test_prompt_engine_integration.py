"""Integration tests for prompt utilities in PromptEngine."""

import pytest

from ai_artist.utils.prompt_engine import PromptEngine


class TestPromptEngineIntegration:
    """Test integration of new prompt utilities into PromptEngine."""

    @pytest.fixture
    def engine(self):
        """Create a prompt engine instance."""
        return PromptEngine()

    def test_emphasis_integration(self, engine):
        """Test that emphasis syntax is processed correctly."""
        # Test with emphasis enabled
        prompt = "a (beautiful:1.5) (landscape:1.2)"
        result = engine.process(prompt, apply_emphasis=True)

        # Should convert to Compel format (uses + for emphasis)
        # (beautiful:1.5) = 1.5 weight = 5 pluses (0.5 / 0.1)
        # (landscape:1.2) = 1.2 weight = 2 pluses but max(1, 2) -> rounds to +
        assert "(beautiful)+++++" in result
        assert "(landscape)+" in result

    def test_emphasis_disabled_by_default(self, engine):
        """Test that emphasis is not applied by default."""
        prompt = "a (beautiful:1.5) landscape"
        result = engine.process(prompt, apply_emphasis=False)

        # Should keep original format
        assert "(beautiful:1.5)" in result

    def test_matrix_generation(self, engine):
        """Test prompt matrix generation."""
        prompt = "a [red|blue] [cat|dog]"
        combinations = engine.process_matrix(prompt)

        assert len(combinations) == 4
        assert "a red cat" in combinations
        assert "a red dog" in combinations
        assert "a blue cat" in combinations
        assert "a blue dog" in combinations

    def test_matrix_with_wildcards(self, engine, tmp_path):
        """Test matrix generation with wildcard integration."""
        # Create a wildcard file
        wildcards_dir = tmp_path / "wildcards"
        wildcards_dir.mkdir()
        (wildcards_dir / "color.txt").write_text("red\\nblue")

        engine_with_wildcards = PromptEngine(wildcards_dir)
        prompt = "a __color__ [cat|dog]"
        combinations = engine_with_wildcards.process_matrix(prompt)

        # Should process wildcards first, then generate matrix
        assert len(combinations) == 2
        assert any("cat" in c for c in combinations)
        assert any("dog" in c for c in combinations)

    def test_combined_features(self, engine):
        """Test combining emphasis and choice syntax."""
        prompt = "(masterpiece:1.5), {vibrant|soft} colors"
        result = engine.process(prompt, apply_emphasis=True)

        # Should have emphasis processed (1.5 = 5 pluses)
        assert "(masterpiece)+++++" in result
        # Should have choice made
        assert "vibrant" in result or "soft" in result

    def test_matrix_with_emphasis(self, engine):
        """Test matrix generation with emphasis syntax."""
        prompt = "a [red|blue] (cat:1.3)"
        combinations = engine.process_matrix(prompt)

        # Matrix should generate combinations (emphasis not processed in matrix by default)
        assert len(combinations) == 2
        assert "a red (cat:1.3)" in combinations or "a blue (cat:1.3)" in combinations

    def test_empty_prompt(self, engine):
        """Test handling of empty prompts."""
        result = engine.process("")
        assert result == ""

    def test_special_characters(self, engine):
        """Test handling of special characters in prompts."""
        prompt = "a cat, (high quality:1.2), detailed"
        result = engine.process(prompt, apply_emphasis=True)

        # Should preserve structure
        assert "cat" in result
        # (high quality:1.2) = 1.2 weight rounds to +
        assert "(high quality)+" in result
        assert "detailed" in result
