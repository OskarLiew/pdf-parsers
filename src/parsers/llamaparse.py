import os
from pathlib import Path

from llama_parse import LlamaParse, ResultType

from src.parsers.base import BasePdfParser


class LlamaParsePdfParser(BasePdfParser):
    def __init__(self, fast: bool = False) -> None:
        super().__init__()
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY", ""),
            result_type=ResultType.TXT if fast else ResultType.MD,
            fast_mode=fast,
        )

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        documents = self.parser.load_data(str(in_path))
        if not documents:
            raise RuntimeError(f"Error parsing {in_path}")
        md = "\n\n".join(doc.text for doc in documents)
        return [(".md", md)]
