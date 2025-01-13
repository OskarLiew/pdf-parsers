PYTHON_SCRIPT="src/main.py"

for parser in docling llama_parse_fast llama_parse llmsherpa marker nougat pdfminer pdfplumber pdfplumber_layout pdftext pdftext_json pymupdf4llm pymupdf pypdf pypdfium2 unstructured_fast unstructured_hires got_ocr2_0 got_ocr2_0_format markitdown; do
  echo "Running parser: $parser"
  python $PYTHON_SCRIPT $parser
done
