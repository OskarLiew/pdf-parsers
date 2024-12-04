from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from src.parsers.base import BasePdfParser


class MarkerPdfParser(BasePdfParser):
    def __init__(self) -> None:
        super().__init__()
        self.converter = PdfConverter(artifact_dict=create_model_dict())

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        rendered = self.converter(str(in_path))
        text, _, _ = text_from_rendered(rendered)
        return [(".md", text)]
