#!/usr/bin/env python3
"""Check Aria's current state and recent creations."""

import json
from datetime import datetime
from pathlib import Path


def check_aria_status():
    """Display Aria's current personality state and recent works."""

    memory_file = Path("data/aria_memory.json")

    if not memory_file.exists():
        print("🎨 Aria hasn't created anything yet!")
        print("Run: python -m ai_artist.main --theme 'your theme'")
        return

    with open(memory_file) as f:
        memory = json.load(f)

    print("\n" + "=" * 60)
    print("✨ ARIA - Autonomous AI Artist")
    print("=" * 60)

    # Basic stats
    total = memory["stats"]["total_created"]
    best_score = memory["stats"]["best_score"]

    print("\n📊 Creative Journey:")
    print(f"   Total Artworks: {total}")
    print(f"   Best Score: {best_score:.3f}")

    if total == 0:
        print("\n🌱 Aria is just beginning her creative journey...")
        return

    # Recent works
    recent = memory["paintings"][-5:]

    print("\n🎨 Recent Creations:")
    for i, painting in enumerate(recent, 1):
        timestamp = datetime.fromisoformat(painting["timestamp"])
        print(f"\n   {i}. {painting['subject']}")
        print(f"      Mood: {painting['mood']}")
        print(f"      Score: {painting['score']:.3f}")
        print(f"      Style: {painting['style']}")
        print(f"      Created: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        if "metadata" in painting and "reflection" in painting["metadata"]:
            print(f"      Reflection: {painting['metadata']['reflection'][:80]}...")

    # Preferences
    fav_subjects = memory["preferences"]["favorite_subjects"]
    fav_styles = memory["preferences"]["favorite_styles"]

    if fav_subjects:
        top_subject = max(fav_subjects.items(), key=lambda x: x[1])
        print(f"\n💭 Favorite Subject: {top_subject[0]} ({top_subject[1]} times)")

    if fav_styles:
        top_style = max(fav_styles.items(), key=lambda x: x[1])
        print(f"   Favorite Style: {top_style[0]} ({top_style[1]} times)")

    print("\n" + "=" * 60)
    print("📖 Full memory at: data/aria_memory.json")
    print("🖼️  Gallery at: gallery/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    check_aria_status()
