from pathlib import Path

from markitdown import MarkItDown

from src.parsers.base import BasePdfParser


class MarkitdownPdfParser(BasePdfParser):
    def __init__(self) -> None:
        super().__init__()
        self.converter = MarkItDown()

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        result = self.converter.convert(str(in_path))
        return [(".txt", result.text_content)]
