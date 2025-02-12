import time
import math
import os
from pathlib import Path
import re

from pdf2image import convert_from_path
from google import genai
from tqdm import tqdm

from src.parsers.base import BasePdfParser

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set")


def with_retry(wait_time, retries):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    t = max(wait_time ** math.sqrt(i + 1), 60)
                    print(f"Function failed. Retrying in {t:.2f} seconds. Error: {exc}")
                    time.sleep(t)

        return wrapper

    return decorator


class Gemini2FlashParser(BasePdfParser):
    def __init__(self) -> None:
        super().__init__()
        self.client = genai.Client()

    @with_retry(wait_time=10, retries=3)
    def generate(self, image):
        return self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Convert this document to markdown. Respond with the output in a code block",
                image,
            ],
        )

    def _parse(self, in_path: Path) -> list[tuple[str, str]]:
        images = convert_from_path(in_path)
        output = []
        for image in tqdm(images, leave=False):
            response = self.generate(image)

            if not response.text:
                raise ValueError("No output from Gemini")

            pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
            result = re.findall(pattern, response.text, re.DOTALL | re.MULTILINE)
            output.extend(result)

        return [(".md", "\n\n".join(output))]
