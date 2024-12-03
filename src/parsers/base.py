from pathlib import Path
from typing import Protocol


class BasePdfParser(Protocol):
    def parse(self, in_path: Path, out_dir: Path) -> None: ...
