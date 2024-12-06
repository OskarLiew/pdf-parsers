from pathlib import Path

import pymupdf

from src.parsers.base import BasePdfParser


class PyMuPdfPdfParser(BasePdfParser):
    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        doc = pymupdf.open(in_path)
        pages_text = []
        for page in doc:
            page_text = page.get_text()
            pages_text.append(page_text)

        return [(".txt", "\n\n".join(pages_text))]
