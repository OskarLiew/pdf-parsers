from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from src.parsers.base import BasePdfParser


class UnstructuredPdfParser(BasePdfParser):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        elements = partition_pdf(str(in_path), languages=["en", "sv"], **self.kwargs)
        pages_text = []
        for element in elements:
            page_text = element.text
            pages_text.append(page_text)

        return [(".txt", "\n\n".join(pages_text))]
