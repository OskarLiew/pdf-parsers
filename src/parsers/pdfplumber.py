from pathlib import Path

import pdfplumber

from src.parsers.base import BasePdfParser


class PdfPlumberPdfParser(BasePdfParser):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        pdf = pdfplumber.open(in_path)
        pages_text = []
        for page in pdf.pages:
            page_text = page.extract_text(**self.kwargs)
            pages_text.append(page_text)
        return [(".txt", "\n\n".join(pages_text))]
