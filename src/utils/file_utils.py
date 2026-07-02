# src/utils/file_utils.py
import yaml
from pathlib import Path

def get_image_paths(data_dir: str) -> list[Path]:
    """
    Recursively finds all .jpg and .png files under data_dir.
    Returns a sorted list of Path objects.
    """
    base = Path(data_dir)
    paths = sorted(
        list(base.glob("**/*.jpg")) +
        list(base.glob("**/*.png"))
    )
    return paths

def ensure_output_dir(base_dir: str, run_name: str) -> Path:
    """
    Creates base_dir/run_name/ if it doesn't exist.
    Returns the Path so the caller can use it immediately.
    """
    out = Path(base_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_yaml(path: str) -> dict:
    """
    Loads a YAML file and returns its contents as a plain dict.
    Raises FileNotFoundError early rather than letting yaml.safe_load fail silently.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with open(p, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Test get_image_paths — point it at any folder on your machine that has images
    # Even your Mac's wallpapers folder works: "/System/Library/Desktop Pictures"
    paths = get_image_paths("/System/Library/Desktop Pictures")
    print(f"Found {len(paths)} images")
    for p in paths[:3]:      # print just the first 3
        print(f"  {p.name}")

    # Test ensure_output_dir
    out = ensure_output_dir("outputs", "test_run")
    print(f"Output dir created: {out}")
    print(f"Exists: {out.exists()}")