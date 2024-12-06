from pathlib import Path

import pypdfium2

from src.parsers.base import BasePdfParser


class Pypdfium2PdfParser(BasePdfParser):
    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        pdf = pypdfium2.PdfDocument(in_path)
        pages_text = []
        for page in pdf:
            textpage = page.get_textpage()
            page_text = textpage.get_text_range()
            pages_text.append(page_text)

        return [(".txt", "\n\n".join(pages_text))]
