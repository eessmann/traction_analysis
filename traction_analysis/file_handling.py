from pathlib import Path
import argparse as ap


def is_empty_dir(path: Path) -> bool:
    """Check if the directory at the given path is empty."""
    return path.is_dir() and not any(path.iterdir())


def dir_path(input_path: str) -> Path:
    """Check if the given path is a valid directory."""
    path = Path(input_path)
    if path.is_dir():
        return path
    else:
        raise ap.ArgumentTypeError(f"readable_dir:{input_path} is not a valid path")
