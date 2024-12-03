from abc import ABC, abstractmethod
from pathlib import Path
from time import time

from src.utils import write_to_file


class BasePdfParser(ABC):
    def __init__(self) -> None:
        self._runtime: float | None = None

    @property
    def runtime(self) -> float:
        if self._runtime is not None:
            return self._runtime
        raise AttributeError("parse not run")

    @abstractmethod
    def _parse(self, in_path: Path) -> list[tuple[str, str]]: ...

    def parse(self, in_path: Path, out_dir: Path) -> None:
        t0 = time()
        result = self._parse(in_path)
        self._runtime = time() - t0
        for suffix, content in result:
            write_to_file(out_dir / in_path.with_suffix(suffix).name, content)
