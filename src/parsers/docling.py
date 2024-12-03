from pathlib import Path

from docling.document_converter import DocumentConverter

from src.parsers.base import BasePdfParser


class DoclingPdfParser(BasePdfParser):
    def __init__(self) -> None:
        super().__init__()
        self.converter = DocumentConverter()

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        result = self.converter.convert(in_path)
        txt = result.document.export_to_text()
        md = result.document.export_to_markdown()
        html = result.document.export_to_html()
        return [(".txt", txt), (".md", md), (".html", html)]
