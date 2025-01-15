from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.parsers.base import BasePdfParser


class DoclingPdfParser(BasePdfParser):
    def __init__(self) -> None:
        super().__init__()
        pdf_options = PdfPipelineOptions(
            ocr_options=EasyOcrOptions(lang=["en", "sv"]),
        )
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        result = self.converter.convert(in_path)
        txt = result.document.export_to_text()
        md = result.document.export_to_markdown()
        html = result.document.export_to_html()
        return [(".txt", txt), (".md", md), (".html", html)]
