"""Aria's mood system - influences her artistic choices."""

import random
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class Mood(str, Enum):
    """Aria's possible moods."""

    CONTEMPLATIVE = "contemplative"
    CHAOTIC = "chaotic"
    MELANCHOLIC = "melancholic"
    ENERGIZED = "energized"
    REBELLIOUS = "rebellious"
    SERENE = "serene"
    RESTLESS = "restless"
    PLAYFUL = "playful"
    INTROSPECTIVE = "introspective"
    BOLD = "bold"


class MoodSystem:
    """Manages Aria's emotional state and how it affects her art."""

    def __init__(self):
        self.current_mood = Mood.CONTEMPLATIVE
        self.energy_level = 0.5  # 0.0 to 1.0
        self.mood_duration = 0
        self.mood_influences = {
            Mood.CONTEMPLATIVE: {
                "styles": ["minimalist", "zen", "abstract", "atmospheric"],
                "colors": ["muted blues", "soft grays", "gentle earth tones"],
                "subjects": ["nature", "meditation", "silence", "space"],
            },
            Mood.CHAOTIC: {
                "styles": ["abstract expressionism", "glitch art", "splatter paint"],
                "colors": ["clashing neons", "explosive reds", "electric yellows"],
                "subjects": ["chaos", "energy", "movement", "destruction"],
            },
            Mood.MELANCHOLIC: {
                "styles": ["impressionist", "muted realism", "somber abstract"],
                "colors": ["deep blues", "dark purples", "shadowy grays"],
                "subjects": ["solitude", "rain", "autumn", "twilight"],
            },
            Mood.ENERGIZED: {
                "styles": ["vibrant pop art", "dynamic composition", "bold colors"],
                "colors": ["bright oranges", "sunny yellows", "vivid greens"],
                "subjects": ["celebration", "life", "movement", "joy"],
            },
            Mood.REBELLIOUS: {
                "styles": ["punk aesthetic", "street art", "provocative"],
                "colors": ["defiant blacks", "anarchic neons", "aggressive reds"],
                "subjects": ["resistance", "protest", "freedom", "challenge"],
            },
            Mood.SERENE: {
                "styles": ["peaceful landscapes", "soft focus", "harmonious"],
                "colors": ["gentle pastels", "calm blues", "peaceful whites"],
                "subjects": ["tranquility", "ocean", "clouds", "harmony"],
            },
            Mood.RESTLESS: {
                "styles": ["fragmented", "layered", "complex"],
                "colors": ["agitated reds", "anxious oranges", "nervous yellows"],
                "subjects": ["searching", "journey", "change", "tension"],
            },
            Mood.PLAYFUL: {
                "styles": ["whimsical", "cartoonish", "fun", "colorful"],
                "colors": ["rainbow hues", "candy colors", "cheerful brights"],
                "subjects": ["childhood", "games", "imagination", "delight"],
            },
            Mood.INTROSPECTIVE: {
                "styles": ["detailed realism", "symbolic", "layered meaning"],
                "colors": ["thoughtful browns", "deep greens", "reflective silvers"],
                "subjects": ["self", "mind", "memory", "dreams"],
            },
            Mood.BOLD: {
                "styles": ["dramatic", "high contrast", "striking"],
                "colors": ["powerful blacks", "intense reds", "commanding golds"],
                "subjects": ["strength", "confidence", "power", "declaration"],
            },
        }

        logger.info("mood_system_initialized", initial_mood=self.current_mood)

    def update_mood(self, external_factors: dict = None) -> Mood:
        """Update Aria's mood based on time, energy, and external factors."""
        self.mood_duration += 1

        # Mood naturally shifts over time
        if self.mood_duration > random.randint(5, 10):
            # Time for a mood change
            self._shift_mood()

        # Energy affects mood intensity
        self.energy_level += random.uniform(-0.1, 0.1)
        self.energy_level = max(0.0, min(1.0, self.energy_level))

        logger.info(
            "mood_updated",
            mood=self.current_mood,
            energy=round(self.energy_level, 2),
            duration=self.mood_duration,
        )

        return self.current_mood

    def _shift_mood(self):
        """Shift to a new mood organically."""
        # Similar moods are more likely to transition
        mood_transitions = {
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

        possible_moods = mood_transitions.get(self.current_mood, list(Mood))
        self.current_mood = random.choice(possible_moods)
        self.mood_duration = 0

        logger.info("mood_shifted", new_mood=self.current_mood)

    def influence_prompt(self, base_prompt: str) -> str:
        """Modify a prompt based on current mood."""
        influences = self.mood_influences[self.current_mood]

        # Add mood-appropriate style
        if random.random() > 0.3:
            style = random.choice(influences["styles"])
            base_prompt = f"{base_prompt}, {style} style"

        # Add mood-appropriate colors
        if random.random() > 0.5:
            colors = random.choice(influences["colors"])
            base_prompt = f"{base_prompt}, {colors}"

        # Add mood descriptor
        mood_descriptors = {
            Mood.CONTEMPLATIVE: "quiet, thoughtful",
            Mood.CHAOTIC: "wild, energetic",
            Mood.MELANCHOLIC: "somber, wistful",
            Mood.ENERGIZED: "vibrant, alive",
            Mood.REBELLIOUS: "defiant, bold",
            Mood.SERENE: "peaceful, calm",
            Mood.RESTLESS: "tense, seeking",
            Mood.PLAYFUL: "fun, joyful",
            Mood.INTROSPECTIVE: "deep, reflective",
            Mood.BOLD: "powerful, striking",
        }

        descriptor = mood_descriptors[self.current_mood]
        base_prompt = f"{base_prompt}, {descriptor} mood"

        logger.info(
            "prompt_influenced_by_mood",
            mood=self.current_mood,
            original_length=len(base_prompt.split(",")),
            energy=round(self.energy_level, 2),
        )

        return base_prompt

    def get_mood_based_subject(self) -> str:
        """Choose a subject based on current mood."""
        influences = self.mood_influences[self.current_mood]
        subject = random.choice(influences["subjects"])

        logger.info(
            "mood_based_subject_chosen", mood=self.current_mood, subject=subject
        )

        return subject

    def describe_feeling(self) -> str:
        """Aria describes how she's feeling."""
        feelings = {
            Mood.CONTEMPLATIVE: "I'm in a contemplative space, seeking deeper meaning.",
            Mood.CHAOTIC: "I feel chaotic energy coursing through me!",
            Mood.MELANCHOLIC: "A melancholic wave has washed over me...",
            Mood.ENERGIZED: "I'm buzzing with energy and excitement!",
            Mood.REBELLIOUS: "I feel rebellious, ready to challenge conventions.",
            Mood.SERENE: "I'm experiencing profound serenity.",
            Mood.RESTLESS: "I'm restless, searching for something new.",
            Mood.PLAYFUL: "I'm feeling playful and whimsical today!",
            Mood.INTROSPECTIVE: "I'm turning inward, exploring my depths.",
            Mood.BOLD: "I feel bold and confident in my vision.",
        }

        return f"{feelings[self.current_mood]} (Energy: {self.energy_level:.0%})"

    def reflect_on_work(self, score: float, subject: str) -> str:
        """Aria reflects on a piece she just created."""
        reflections = {
            Mood.CONTEMPLATIVE: [
                f"This {subject} piece emerged from a place of deep thought.",
                f"I pondered long on this {subject} before bringing it to life.",
                f"In contemplating {subject}, I found unexpected depth.",
            ],
            Mood.CHAOTIC: [
                f"This {subject} exploded from my chaotic energy!",
                f"The chaos within me manifested in this wild {subject}.",
                f"I threw everything at this {subject} - pure creative chaos!",
            ],
            Mood.MELANCHOLIC: [
                f"A melancholic mood washed over this {subject}.",
                f"I poured my sadness into this {subject}.",
                f"The weight of melancholy shaped this {subject}.",
            ],
            Mood.ENERGIZED: [
                f"So much energy went into this {subject}!",
                f"I was buzzing with excitement creating this {subject}!",
                f"This {subject} radiates the energy I felt!",
            ],
            Mood.REBELLIOUS: [
                f"This {subject} defies expectations.",
                f"I challenged conventions with this {subject}.",
                f"A rebellious spirit drove this {subject}.",
            ],
            Mood.SERENE: [
                f"Serenity flowed through this {subject}.",
                f"I found peace in creating this {subject}.",
                f"This {subject} emerged from a place of calm.",
            ],
            Mood.RESTLESS: [
                f"My restless energy pushed this {subject} forward.",
                f"I couldn't sit still while making this {subject}.",
                f"Restlessness fueled this {subject}.",
            ],
            Mood.PLAYFUL: [
                f"I had so much fun playing with this {subject}!",
                f"Whimsy and play guided this {subject}.",
                f"This {subject} is pure playful joy!",
            ],
            Mood.INTROSPECTIVE: [
                f"Looking inward revealed this {subject}.",
                f"Deep introspection birthed this {subject}.",
                f"I explored my inner world through this {subject}.",
            ],
            Mood.BOLD: [
                f"I made a bold statement with this {subject}!",
                f"Confidence drove this striking {subject}.",
                f"This {subject} is unapologetically bold.",
            ],
        }

        base_reflection = random.choice(reflections[self.current_mood])

        # Add score-based reflection
        if score >= 0.7:
            base_reflection += " I'm really pleased with how it turned out."
        elif score >= 0.5:
            base_reflection += " It's interesting, though not my best work."
        else:
            base_reflection += " It didn't quite capture what I envisioned, but that's part of the journey."

        return base_reflection

    def get_mood_colors(self) -> list[str]:
        """Get the color palette for current mood."""
        influences = self.mood_influences[self.current_mood]
        return influences["colors"]
