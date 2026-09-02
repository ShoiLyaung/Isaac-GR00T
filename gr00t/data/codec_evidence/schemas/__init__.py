"""Branch-owned JSON schemas for codec-evidence artifacts."""

from importlib.resources import files
from pathlib import Path


def schema_path(name: str) -> Path:
    """Return the local path of one explicitly named bundled schema."""

    if not name or "/" in name or "\\" in name or not name.endswith(".schema.json"):
        raise ValueError("Schema name must be one local *.schema.json filename")
    resource = files(__package__).joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(name)
    return Path(str(resource))


__all__ = ["schema_path"]
