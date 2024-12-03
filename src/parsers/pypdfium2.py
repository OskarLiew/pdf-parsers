from pathlib import Path

from src.parsers.base import BasePdfParser


class PdfParser(BasePdfParser):
    def parse(self, in_path: Path, out_dir: Path) -> None: ...
