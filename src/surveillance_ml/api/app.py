"""Small authenticated FastAPI service for ONNX scene classification."""

import asyncio
import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from surveillance_ml.api.middleware import RequestBodyLimitMiddleware
from surveillance_ml.inference.onnx_classifier import OnnxClassifier

SUPPORTED_CONTENT_TYPES = {"image/bmp", "image/jpeg", "image/png", "image/webp"}
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000


def create_app(
    model_path: Path | None = None,
    manifest_path: Path | None = None,
    api_key: str | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    """Create the application; model initialization occurs in its lifespan."""
    resolved_key = api_key if api_key is not None else os.environ.get("SURVEILLANCE_API_KEY")
    if not resolved_key:
        raise ValueError("SURVEILLANCE_API_KEY must contain at least 32 characters")
    try:
        key_bytes = resolved_key.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("SURVEILLANCE_API_KEY must contain only ASCII characters") from error
    if len(key_bytes) < 32:
        raise ValueError("SURVEILLANCE_API_KEY must contain at least 32 characters")
    resolved_model = model_path or Path(os.environ.get("SURVEILLANCE_MODEL", "model.onnx"))
    resolved_manifest = manifest_path or Path(
        os.environ.get("SURVEILLANCE_MANIFEST", "manifest.json")
    )
    upload_limit = (
        max_upload_bytes
        if max_upload_bytes is not None
        else int(os.environ.get("SURVEILLANCE_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES))
    )
    if upload_limit < 1:
        raise ValueError("Maximum upload size must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.classifier = OnnxClassifier(resolved_model, resolved_manifest)
        try:
            yield
        finally:
            app.state.classifier = None

    app = FastAPI(
        title="Surveillance AI Training Inference",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=upload_limit + 64 * 1024,
    )

    async def require_api_key(
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        try:
            supplied_key = x_api_key.encode("ascii") if x_api_key is not None else b""
        except UnicodeEncodeError:
            supplied_key = b""
        if not hmac.compare_digest(supplied_key, key_bytes):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    async def health() -> dict[str, str]:
        classifier = getattr(app.state, "classifier", None)
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {
            "status": "ok",
            "architecture": classifier.manifest.architecture,
            "model_sha256": classifier.manifest.model_sha256,
        }

    @app.post("/predict", dependencies=[Depends(require_api_key)])
    async def predict(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        if file.content_type not in SUPPORTED_CONTENT_TYPES:
            await file.close()
            raise HTTPException(status_code=415, detail="Unsupported image content type")
        try:
            content = await file.read(upload_limit + 1)
        finally:
            await file.close()
        if len(content) > upload_limit:
            raise HTTPException(status_code=413, detail="Image exceeds the upload size limit")
        try:
            with Image.open(BytesIO(content)) as image:
                if image.width * image.height > DEFAULT_MAX_IMAGE_PIXELS:
                    raise OverflowError("Image exceeds the decoded pixel limit")
                image.load()
                if image.format not in {"BMP", "JPEG", "PNG", "WEBP"}:
                    raise ValueError("Unsupported image format")
                rgb_image = image.convert("RGB")
        except (
            OSError,
            ValueError,
            OverflowError,
            Image.DecompressionBombError,
            UnidentifiedImageError,
        ) as error:
            status_code = 413 if isinstance(error, OverflowError) else 400
            detail = (
                "Image exceeds the decoded pixel limit"
                if status_code == 413
                else "Invalid or corrupt image"
            )
            raise HTTPException(status_code=status_code, detail=detail) from error
        classifier = getattr(app.state, "classifier", None)
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        try:
            prediction = await asyncio.to_thread(classifier.predict_image, rgb_image)
        finally:
            rgb_image.close()
        return {
            "class_name": prediction.class_name,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "model_sha256": classifier.manifest.model_sha256,
        }

    return app
