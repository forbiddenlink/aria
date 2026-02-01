"""Prompt collections for AI Artist image generation.

This module provides centralized access to all prompt collections used for
generating artwork. Each collection focuses on different themes and styles.

Collections:
- artistic: Creative styles and abstract concepts (impressionist, cubist, etc.)
- artistic2: Modern art movements and experimental techniques
- expanded: Fresh creative prompts (cinematic, emotional, nature)
- ultimate: Comprehensive diverse prompts across many themes

Usage:
    from ai_artist.prompts import get_collection_prompts, get_all_collections

    # Get prompts for a specific collection
    prompts = get_collection_prompts("artistic")

    # Get all collections merged
    all_prompts = get_collection_prompts("all")

    # List available collections
    collections = get_all_collections()
"""

from .artistic import ARTISTIC_PROMPTS
from .artistic2 import ARTISTIC_PROMPTS_2
from .expanded import EXPANDED_PROMPTS
from .ultimate import ULTIMATE_PROMPTS

# Collection registry mapping names to prompt dictionaries
COLLECTIONS = {
    "artistic": ARTISTIC_PROMPTS,
    "artistic2": ARTISTIC_PROMPTS_2,
    "expanded": EXPANDED_PROMPTS,
    "ultimate": ULTIMATE_PROMPTS,
}

# Collection metadata for display
COLLECTION_METADATA = {
    "artistic": {
        "name": "Artistic Styles Collection",
        "description": "Creative styles and abstract concepts - impressionist, cubist, surrealist, etc.",
    },
    "artistic2": {
        "name": "Artistic Styles Collection II",
        "description": "Modern art movements and experimental techniques - renaissance, photo-realism, op-art, etc.",
    },
    "expanded": {
        "name": "Expanded Collection",
        "description": "Fresh creative prompts - cinematic moments, emotional portraits, nature spectacles.",
    },
    "ultimate": {
        "name": "Ultimate Collection",
        "description": "Comprehensive diverse prompts across many themes - cosmic, mythological, retro, fantasy.",
    },
}


def get_collection_prompts(collection_name: str) -> dict[str, list[str]]:
    """Get prompts for a specific collection.

    Args:
        collection_name: Name of the collection ('artistic', 'artistic2',
                        'expanded', 'ultimate', or 'all')

    Returns:
        Dictionary mapping category names to lists of prompts

    Raises:
        ValueError: If collection_name is not recognized
    """
    if collection_name == "all":
        # Merge all collections
        merged: dict[str, list[str]] = {}
        for prompts in COLLECTIONS.values():
            for category, prompt_list in prompts.items():
                if category in merged:
                    # Prefix to avoid collisions
                    merged[category].extend(prompt_list)
                else:
                    merged[category] = list(prompt_list)
        return merged

    if collection_name not in COLLECTIONS:
        valid = ", ".join(sorted(COLLECTIONS.keys()))
        raise ValueError(
            f"Unknown collection '{collection_name}'. Valid options: {valid}, all"
        )

    return COLLECTIONS[collection_name]


def get_all_collections() -> dict[str, dict[str, list[str]]]:
    """Get all available collections.

    Returns:
        Dictionary mapping collection names to their prompt dictionaries
    """
    return COLLECTIONS.copy()


def get_collection_names() -> list[str]:
    """Get list of available collection names.

    Returns:
        List of collection names
    """
    return list(COLLECTIONS.keys())


def get_collection_info(collection_name: str | None = None) -> dict:
    """Get metadata about collections.

    Args:
        collection_name: Optional specific collection name. If None, returns all.

    Returns:
        Dictionary with collection metadata including name, description,
        number of categories, and total prompts.
    """
    if collection_name:
        if collection_name not in COLLECTIONS:
            valid = ", ".join(sorted(COLLECTIONS.keys()))
            raise ValueError(
                f"Unknown collection '{collection_name}'. Valid options: {valid}"
            )
        prompts = COLLECTIONS[collection_name]
        meta = COLLECTION_METADATA[collection_name]
        return {
            "name": meta["name"],
            "description": meta["description"],
            "categories": len(prompts),
            "total_prompts": sum(len(p) for p in prompts.values()),
        }

    # Return info for all collections
    result = {}
    for name, prompts in COLLECTIONS.items():
        meta = COLLECTION_METADATA[name]
        result[name] = {
            "name": meta["name"],
            "description": meta["description"],
            "categories": len(prompts),
            "total_prompts": sum(len(p) for p in prompts.values()),
        }
    return result


def count_prompts(collection_name: str | None = None) -> int:
    """Count total prompts in a collection or all collections.

    Args:
        collection_name: Optional collection name. If None, counts all.

    Returns:
        Total number of prompts
    """
    if collection_name:
        prompts = get_collection_prompts(collection_name)
        return sum(len(p) for p in prompts.values())

    total = 0
    for prompts in COLLECTIONS.values():
        total += sum(len(p) for p in prompts.values())
    return total


__all__ = [
    "ARTISTIC_PROMPTS",
    "ARTISTIC_PROMPTS_2",
    "EXPANDED_PROMPTS",
    "ULTIMATE_PROMPTS",
    "COLLECTIONS",
    "COLLECTION_METADATA",
    "get_collection_prompts",
    "get_all_collections",
    "get_collection_names",
    "get_collection_info",
    "count_prompts",
]
