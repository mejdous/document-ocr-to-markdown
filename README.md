# document-ocr-to-markdown

A lightweight Python REST API that converts **PDF** or **image** files to **Markdown** using OCR — no LLM, no cloud dependency, fully in-memory.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/diag` | Dependency & runtime diagnostics |
| `POST` | `/pdf_to_markdown` | PDF → Markdown (raw body = PDF bytes) |
| `POST` | `/image_to_markdown` | Image (PNG/JPG/JPEG) → Markdown (raw body or `multipart/form-data`) |

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The interactive docs (Swagger UI) are available at `http://localhost:8000/docs`.

## Usage examples

### Diagnostics

```bash
curl http://localhost:8000/diag
```

### PDF → Markdown

```bash
curl -X POST http://localhost:8000/pdf_to_markdown \
     --data-binary @document.pdf \
     -H "Content-Type: application/pdf"
```

### Image → Markdown (raw body)

```bash
curl -X POST http://localhost:8000/image_to_markdown \
     --data-binary @invoice.png \
     -H "Content-Type: image/png"
```

### Image → Markdown (multipart/form-data)

```bash
curl -X POST http://localhost:8000/image_to_markdown \
     -F "file=@invoice.png"
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` / `uvicorn` | Web framework & ASGI server |
| `pymupdf` + `pymupdf4llm` | PDF parsing and Markdown extraction |
| `rapidocr-onnxruntime` | OCR engine (ONNX, no system binaries required) |
| `Pillow` / `numpy` | Image decoding and array manipulation |