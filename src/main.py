from functools import cache
from pathlib import Path

import pypdf
from parsers.pypdfium2 import Pypdfium2PdfParser
from parsers.unstructured import UnstructuredPdfParser
from pydantic import BaseModel
from tqdm import tqdm
from utils import write_to_file

from src.parsers.base import BasePdfParser
from src.parsers.docling import DoclingPdfParser
from src.parsers.llamaparse import LlamaParsePdfParser
from src.parsers.llmsherpa import LlmsherpaPdfParser
from src.parsers.marker import MarkerPdfParser
from src.parsers.nougat import NougatPdfParser
from src.parsers.pdfminer import PdfMinerPdfParser
from src.parsers.pdfplumber import PdfPlumberPdfParser
from src.parsers.pdftext import PdftextPdfParser
from src.parsers.pymupdf4llm import PyMuPdf4llmPdfParser
from src.parsers.pymypdf import PyMuPdfPdfParser
from src.parsers.pypdf import PyPDFParser

REPO_ROOT = Path(__name__).parent.parent
DOC_DIR = REPO_ROOT / "docs"
OUT_DIR = REPO_ROOT / "out"

if not OUT_DIR.exists():
    OUT_DIR.mkdir()


class LoadedFileInfo(BaseModel):
    name: str
    n_pages: int
    load_time: float
    load_time_per_page: float


class ParserInfo(BaseModel):
    name: str
    files_loaded: list[LoadedFileInfo]
    avg_load_time_per_page: float | None = None


def main():
    parsers: dict[str, BasePdfParser] = {
        "docling": DoclingPdfParser(),
        "llama-parse-fast": LlamaParsePdfParser(fast=True),
        "llama-parse": LlamaParsePdfParser(fast=False),
        "llmsherpa": LlmsherpaPdfParser(),
        "marker": MarkerPdfParser(),
        "nougat": NougatPdfParser(),
        "pdfminer": PdfMinerPdfParser(),
        "pdfplumber": PdfPlumberPdfParser(),
        "pdfplumber-layout": PdfPlumberPdfParser(layout=True),
        "pdftext": PdftextPdfParser("txt"),
        "pdftext-json": PdftextPdfParser("json"),
        "pymupdf4llm": PyMuPdf4llmPdfParser(),
        "pymupdf": PyMuPdfPdfParser(),
        "pypdf": PyPDFParser(),
        "pypdfium2": Pypdfium2PdfParser(),
        "unstructured-fast": UnstructuredPdfParser(strategy="fast"),
        "unstructured-hires": UnstructuredPdfParser(strategy="hi_res"),
    }

    for parser_name, parser in parsers.items():
        print(f"Evaluating parser: {parser_name}")

        parser_out_dir = OUT_DIR / parser_name
        parser_out_dir.mkdir(exist_ok=True)
        parser_info = ParserInfo(name=parser_name, files_loaded=[])

        for doc_path in tqdm(list(DOC_DIR.iterdir()), desc="Processing documents"):
            parser.parse(doc_path, parser_out_dir)

            n_pages = get_n_pages(doc_path)
            loaded_file_info = LoadedFileInfo(
                name=doc_path.name,
                n_pages=n_pages,
                load_time=parser.runtime,
                load_time_per_page=parser.runtime / n_pages,
            )
            parser_info.files_loaded.append(loaded_file_info)

        parser_info.avg_load_time_per_page = sum(
            [file_info.load_time_per_page for file_info in parser_info.files_loaded]
        ) / len(parser_info.files_loaded)

        write_to_file(
            parser_out_dir / "parser_info.json", parser_info.model_dump_json(indent=4)
        )


@cache
def get_n_pages(path: Path) -> int:
    pdf_reader = pypdf.PdfReader(path)
    return len(pdf_reader.pages)


if __name__ == "__main__":
    main()
