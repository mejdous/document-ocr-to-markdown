"""
document-ocr-to-markdown
Python FastAPI — converts PDF or image to Markdown.

Endpoints
---------
GET  /diag               Dependency + environment diagnostics
POST /pdf_to_markdown    PDF  → Markdown  (body = raw PDF bytes)
POST /image_to_markdown  Image → Markdown (body = raw image bytes, or multipart/form-data)
"""

import io
import logging
import sys
import traceback

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="document-ocr-to-markdown",
    description="Convert PDF or image to Markdown via OCR.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# GET /diag
# ---------------------------------------------------------------------------

@app.get("/diag", response_class=PlainTextResponse)
async def diag() -> str:
    """Diagnostic: check dependencies and runtime environment."""
    lines = [f"Python : {sys.version}", ""]

    for pkg in ["numpy", "PIL", "cv2", "rapidocr_onnxruntime", "pymupdf", "pymupdf4llm"]:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            lines.append(f"✅ {pkg} {version}")
        except Exception as exc:
            lines.append(f"❌ {pkg} — {exc}")

    # Functional RapidOCR test on a tiny white image
    lines.append("")
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        dummy = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result, _ = ocr(dummy)
        lines.append("✅ RapidOCR functional")
    except Exception as exc:
        lines.append(f"❌ RapidOCR functional — {exc}")
        lines.append(traceback.format_exc())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# POST /pdf_to_markdown
# ---------------------------------------------------------------------------

@app.post("/pdf_to_markdown")
async def pdf_to_markdown(request: Request) -> Response:
    """
    Accept a PDF file as the raw request body and return the extracted Markdown.
    Uses pymupdf4llm — fully in-memory, no LLM, no disk writes.
    """
    logger.info("pdf_to_markdown: request received.")

    pdf_bytes = await request.body()
    if not pdf_bytes:
        return PlainTextResponse(
            "Aucun contenu PDF fourni dans le corps de la requête.",
            status_code=400,
        )

    try:
        import pymupdf4llm  # noqa: PLC0415
    except ImportError as exc:
        logger.error("pymupdf4llm not available: %s", exc)
        return PlainTextResponse(
            "La dépendance pymupdf4llm n'est pas installée sur le serveur.",
            status_code=500,
        )

    try:
        import pymupdf  # noqa: PLC0415

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        markdown_text = pymupdf4llm.to_markdown(doc)

        return Response(
            content=markdown_text,
            status_code=200,
            media_type="text/markdown; charset=utf-8",
        )
    except Exception as exc:
        logger.exception("Error converting PDF → Markdown")
        return PlainTextResponse(
            f"Erreur lors de la conversion : {exc}",
            status_code=500,
        )


# ---------------------------------------------------------------------------
# POST /image_to_markdown
# ---------------------------------------------------------------------------

@app.post("/image_to_markdown")
async def image_to_markdown(request: Request) -> Response:
    """
    Accept an image (PNG, JPG, JPEG) as the raw request body or as a
    multipart/form-data upload and return structured Markdown (tables + text)
    extracted without any LLM via RapidOCR (ONNX, no system binaries required).
    """
    try:
        return await _image_to_markdown_impl(request)
    except BaseException as exc:
        logger.exception("Uncaught error in image_to_markdown")
        return PlainTextResponse(
            f"Erreur non capturée : {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            status_code=500,
        )


async def _image_to_markdown_impl(request: Request) -> Response:
    logger.info("image_to_markdown: request received.")


    content_type = request.headers.get("content-type", "")

    # Multipart/form-data (--form) or raw body (--data-binary)
    img_bytes: bytes | None = None
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            logger.info(f"Champs reçus dans le form: {list(form.keys())}")
            for field_name, field in form.items():
                if hasattr(field, "read"):
                    img_bytes = await field.read()
                    logger.info(f"Champ lu: {field_name}, taille: {len(img_bytes)} octets")
                    logger.info(f"Premiers octets: {img_bytes[:16].hex()}")
                    break
        except Exception as exc:
            logger.error(f"Erreur lors de la lecture du form: {exc}")
            img_bytes = None

    if not img_bytes:
        img_bytes = await request.body() or None
        if img_bytes:
            logger.info(f"Lecture du body brut, taille: {len(img_bytes)} octets")
            logger.info(f"Premiers octets: {img_bytes[:16].hex()}")

    if not img_bytes:
        logger.warning("Aucun contenu image fourni dans la requête.")
        return PlainTextResponse(
            "Aucun contenu image fourni dans le corps de la requête.",
            status_code=400,
        )

    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
    except ImportError as exc:
        logger.error("rapidocr-onnxruntime not available: %s", exc)
        return PlainTextResponse(
            "La dépendance rapidocr-onnxruntime n'est pas installée sur le serveur.",
            status_code=500,
        )

    try:
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

        # Redimensionner si l'image est trop grande (évite timeout/OOM sur Render)
        MAX_DIM = 2048
        w, h = pil_img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            logger.info(f"Image redimensionnée: {w}x{h} → {new_w}x{new_h}")

        img_array = np.array(pil_img)

        ocr = RapidOCR()
        result, _ = ocr(img_array)

        if not result:
            return Response(
                content="Aucun contenu détecté dans l'image.",
                status_code=200,
                media_type="text/markdown; charset=utf-8",
            )

        # Each element: [bbox_points, text, score]
        # bbox_points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        # Group words into lines by vertical centroid proximity
        items: list[dict] = []
        for bbox, text, _score in result:
            ys = [pt[1] for pt in bbox]
            xs = [pt[0] for pt in bbox]
            cy = sum(ys) / len(ys)
            cx = sum(xs) / len(xs)
            h = max(ys) - min(ys)
            items.append({"text": text, "cx": cx, "cy": cy, "h": h})

        items.sort(key=lambda i: i["cy"])

        # Grouping threshold = half the median character height
        heights = sorted(i["h"] for i in items)
        median_h = heights[len(heights) // 2] if heights else 10
        threshold = median_h * 0.6

        lines: list[list[dict]] = []
        for item in items:
            if lines and abs(item["cy"] - lines[-1][0]["cy"]) <= threshold:
                lines[-1].append(item)
            else:
                lines.append([item])

        # Sort each line left-to-right
        for line in lines:
            line.sort(key=lambda i: i["cx"])

        # Detect table-like structure (≥ 2 columns, repeated column count)
        col_counts = [len(line) for line in lines if len(line) > 1]
        is_table = len(col_counts) >= 3 and len(set(col_counts)) <= 3

        if is_table:
            n_cols = max(set(col_counts), key=col_counts.count)
            rows_text = []
            for line in lines:
                cells = [item["text"] for item in line]
                cells = (cells + [""] * n_cols)[:n_cols]
                rows_text.append(cells)

            header = rows_text[0]
            separator = ["---"] * n_cols
            body = rows_text[1:]

            def md_row(cells: list[str]) -> str:
                return "| " + " | ".join(cells) + " |"

            md_lines = [md_row(header), md_row(separator)] + [md_row(r) for r in body]
            markdown_text = "\n".join(md_lines)
        else:
            markdown_text = "\n".join(
                "  ".join(item["text"] for item in line) for line in lines
            )

        return Response(
            content=markdown_text,
            status_code=200,
            media_type="text/markdown; charset=utf-8",
        )

    except Exception as exc:
        logger.exception("Error converting image → Markdown")
        return PlainTextResponse(
            f"Erreur lors de la conversion : {exc}",
            status_code=500,
        )
