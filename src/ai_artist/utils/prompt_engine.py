"""Prompt processing engine with wildcard support."""

import random
import re
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)


class PromptEngine:
    """Handles dynamic prompt generation with wildcards and randomization.

    Supports:
    - Choice syntax: {option A|option B|option C}
    - Wildcards: __wildcard_name__ (loads from config/wildcards/wildcard_name.txt)
    """

    def __init__(self, wildcards_dir: Path = Path("config/wildcards")):
        self.wildcards_dir = wildcards_dir
        self.wildcards: dict[str, list[str]] = {}
        self._load_wildcards()

    def reload(self):
        """Reload all wildcards from disk."""
        self.wildcards.clear()
        self._load_wildcards()

    def _load_wildcards(self):
        """Load all wildcard files from directory."""
        if not self.wildcards_dir.exists():
            logger.warning("wildcards_dir_not_found", path=str(self.wildcards_dir))
            return

        for file_path in self.wildcards_dir.glob("*.txt"):
            key = f"__{file_path.stem}__"
            try:
                with open(file_path, encoding="utf-8") as f:
                    lines = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                if lines:
                    self.wildcards[key] = lines
                    logger.debug("wildcard_loaded", key=key, count=len(lines))
            except Exception as e:
                logger.error("wildcard_load_failed", file=file_path.name, error=str(e))

    def _process_choices(self, text: str) -> str:
        """Process {a|b|c} syntax."""

        def replace(match):
            choices = match.group(1).split("|")
            return random.choice(choices).strip()

        # Recursively replace choices (in case of nested choices)
        while "{" in text and "}" in text:
            new_text = re.sub(r"\{([^{}]+)\}", replace, text)
            if new_text == text:  # No changes made, break to avoid infinite loop
                break
            text = new_text

        return text

    def _process_wildcards(self, text: str) -> str:
        """Replace __wildcard__ with random line from file."""
        for key, values in self.wildcards.items():
            if key in text:
                # Replace ALL occurrences of this wildcard
                # Use a loop to support multiple distinct random choices for same wildcard
                # e.g. "__color__ and __color__" -> "red and blue"
                while key in text:
                    replacement = random.choice(values)
                    text = text.replace(key, replacement, 1)
        return text

    def process(self, prompt: str) -> str:
        """Process a prompt with all dynamic features."""
        # First process wildcards (which might contain choices)
        prompt = self._process_wildcards(prompt)
        # Then process choices
        prompt = self._process_choices(prompt)

        # Clean up double spaces and commas
        prompt = re.sub(r"\s+", " ", prompt).strip()
        prompt = re.sub(r"\s*,\s*", ", ", prompt)
        prompt = re.sub(r",,+", ",", prompt)

        return prompt
