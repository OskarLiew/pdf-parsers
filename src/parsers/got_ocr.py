from pathlib import Path
import tempfile
from typing import Literal

from pdf2image import convert_from_path
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.parsers.base import BasePdfParser


class GotOcrPdfParser(BasePdfParser):
    def __init__(self, mode: Literal["ocr", "format"] = "ocr") -> None:
        super().__init__()
        self.model_name = "stepfun-ai/GOT-OCR2_0"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cuda",
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model = self.model.eval().cuda()
        self.mode = mode
        self.out_type = ".txt" if mode == "ocr" else ".tex"

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        pages_out = []
        with tempfile.TemporaryDirectory("pdf_pages") as tmpdir:
            convert_from_path(in_path, output_folder=tmpdir)
            for image_path in tqdm(list(Path(tmpdir).iterdir()), desc="OCR:ing"):
                result = self.model.chat(
                    self.tokenizer, str(image_path), ocr_type=self.mode
                )
                pages_out.append(result)

        return [(self.out_type, "\n\n".join(pages_out))]
