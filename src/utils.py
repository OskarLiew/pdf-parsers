from pathlib import Path


def write_to_file(path: Path, s: str) -> None:
    with open(path, "w") as f:
        f.write(s)
