import json
from pathlib import Path
from typing import Literal

from pdftext.extraction import dictionary_output, plain_text_output

from src.parsers.base import BasePdfParser


class PdftextPdfParser(BasePdfParser):
    def __init__(self, output: Literal["json", "txt"] = "txt") -> None:
        super().__init__()
        self.output = output
        self.extractor = {
            "txt": plain_text_output,
            "json": lambda path: json.dumps(
                dictionary_output(path), ensure_ascii=False, indent=4
            ),
        }[output]

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        content = self.extractor(in_path)
        return [(f".{self.output}", content)]
