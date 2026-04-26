"""Static asset path resolution for local and installed runs."""

from pathlib import Path

DEFAULT_STATIC_DIR = "frontend"
SOURCE_STATIC_DIR = Path(__file__).parent.parent / "frontend"
PACKAGED_STATIC_DIR = Path(__file__).parent / "frontend"


def resolve_static_dir(static_dir: str | Path = DEFAULT_STATIC_DIR) -> Path:
    """Resolve frontend assets, preferring local dev files over packaged files."""
    requested = Path(static_dir)
    resolved = requested if requested.is_absolute() else Path.cwd() / requested

    if resolved.exists():
        return resolved

    if requested == Path(DEFAULT_STATIC_DIR):
        for candidate in (SOURCE_STATIC_DIR, PACKAGED_STATIC_DIR):
            if candidate.exists():
                return candidate

    return resolved
