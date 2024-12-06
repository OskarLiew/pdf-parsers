from pathlib import Path

import pymupdf4llm

from src.parsers.base import BasePdfParser


class PyMuPdf4llmPdfParser(BasePdfParser):
    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        text = pymupdf4llm.to_markdown(in_path)
        return [(".md", text)]
