from pathlib import Path

from pdfminer.high_level import extract_text

from src.parsers.base import BasePdfParser


class PdfMinerPdfParser(BasePdfParser):
    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        text = extract_text(in_path)
        return [(".txt", text)]
