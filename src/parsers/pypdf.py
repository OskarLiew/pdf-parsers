from pathlib import Path

from pypdf import PdfReader

from src.parsers.base import BasePdfParser


class PyPDFParser(BasePdfParser):
    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        reader = PdfReader(in_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            pages_text.append(page_text)

        return [(".txt", "\n\n".join(pages_text))]
