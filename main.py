"""
main.py
========
FastAPI application entry point for MedExplain.

Routes
------
GET  /                      -> serves static/index.html
POST /api/analyze-text      -> simplifies raw clinical text (Module 1: NLP)
POST /api/analyze-xray      -> runs DenseNet-121 + Grad-CAM on an uploaded
                                image and optionally simplifies accompanying
                                clinical findings text (Modules 1 + 2)
POST /api/analyze-blood     -> evaluates blood biomarker JSON payload
                                (Module 3: tabular rule classifier)
GET  /api/health            -> simple liveness/readiness probe

Run locally with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ml_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medexplain")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="MedExplain API",
    description=(
        "Multi-Modal Clinical Intelligence System with Neural Text "
        "Simplification and Visual Pathological Grounding. "
        "Educational demo — not a certified diagnostic device."
    ),
    version="1.0.0",
)

# CORS: permissive for local development / grading purposes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
ALLOWED_DOCUMENT_CONTENT_TYPES = ALLOWED_IMAGE_CONTENT_TYPES | {"application/pdf"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DOCUMENT_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------
class TextSimplifyRequest(BaseModel):
    clinical_text: str = Field(..., min_length=1, description="Raw clinical note text")


class TextSimplifyResponse(BaseModel):
    simplified_text: str


class BloodMetricsRequest(BaseModel):
    hemoglobin: Optional[float] = Field(default=None, description="g/dL")
    wbc: Optional[float] = Field(default=None, description="x10^3/uL")
    platelets: Optional[float] = Field(default=None, description="x10^3/uL")
    glucose: Optional[float] = Field(default=None, description="mg/dL, fasting")


class BiomarkerResult(BaseModel):
    key: str
    label: str
    value: float
    unit: str
    normal_range: str
    status: str
    insight: str


class BloodAnalysisResponse(BaseModel):
    results: list[BiomarkerResult]


class XrayAnalysisResponse(BaseModel):
    simplified_text: str
    heatmap_image_base64: str
    top_features: list[dict]


class PrescriptionAnalysisResponse(BaseModel):
    raw_text: str
    identified_medications: list[dict]
    plain_summary: str


class SymptomCheckRequest(BaseModel):
    symptom: str = Field(..., min_length=1, description="Free-text symptom, e.g. 'headache'")


class SymptomCheckResponse(BaseModel):
    matched: bool
    query: str
    matched_symptom: Optional[str]
    common_causes: list[str]
    self_care: list[str]
    red_flags: list[str]
    disclaimer: str
    message: Optional[str]


class BloodDocumentAnalysisResponse(BaseModel):
    extracted_values: dict
    results: list[BiomarkerResult]


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok", "device": str(ml_engine.DEVICE)}


@app.post("/api/analyze-text", response_model=TextSimplifyResponse)
def analyze_text(payload: TextSimplifyRequest) -> TextSimplifyResponse:
    """Module 1 (NLP): simplify a clinical note into plain English."""
    try:
        simplified = ml_engine.simplify_clinical_text(payload.clinical_text)
        return TextSimplifyResponse(simplified_text=simplified)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Text simplification failed")
        raise HTTPException(status_code=500, detail=f"Text simplification failed: {exc}") from exc


@app.post("/api/analyze-xray", response_model=XrayAnalysisResponse)
async def analyze_xray(
    image: UploadFile = File(..., description="Chest X-ray image file (PNG/JPG)"),
    clinical_text: str = Form(default="", description="Optional accompanying radiology note"),
) -> XrayAnalysisResponse:
    """Modules 1 + 2: run Grad-CAM visual grounding and simplify any accompanying note."""
    if image.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{image.content_type}'. Please upload a PNG or JPEG image.",
        )

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded image exceeds the 10 MB size limit.")

    try:
        heatmap_b64, top_features = ml_engine.generate_xray_cam(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Grad-CAM generation failed")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}") from exc

    try:
        simplified_text = ml_engine.simplify_clinical_text(clinical_text) if clinical_text.strip() else (
            "No accompanying clinical note was provided. The heatmap on the right highlights "
            "the image regions that most influenced the model's output, but a radiologist's "
            "written findings are needed for a full plain-English explanation."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Text simplification failed during xray analysis")
        raise HTTPException(status_code=500, detail=f"Text simplification failed: {exc}") from exc

    return XrayAnalysisResponse(
        simplified_text=simplified_text,
        heatmap_image_base64=heatmap_b64,
        top_features=top_features,
    )


@app.post("/api/analyze-blood", response_model=BloodAnalysisResponse)
def analyze_blood(payload: BloodMetricsRequest) -> BloodAnalysisResponse:
    """Module 3: evaluate blood biomarkers against clinical reference ranges."""
    metrics = payload.model_dump(exclude_none=True)
    if not metrics:
        raise HTTPException(status_code=400, detail="At least one blood metric must be provided.")
    try:
        results = ml_engine.evaluate_blood_biomarkers(metrics)
        return BloodAnalysisResponse(results=results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Blood biomarker evaluation failed")
        raise HTTPException(status_code=500, detail=f"Blood analysis failed: {exc}") from exc


@app.post("/api/analyze-blood-document", response_model=BloodDocumentAnalysisResponse)
async def analyze_blood_document(
    file: UploadFile = File(..., description="Photo or PDF of a blood lab report"),
) -> BloodDocumentAnalysisResponse:
    """Module 6: OCR/parse an uploaded blood report image or PDF, then evaluate any values found."""
    if file.content_type not in ALLOWED_DOCUMENT_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a PNG, JPG, or PDF.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds the 15 MB size limit.")

    try:
        outcome = ml_engine.analyze_blood_document(file_bytes, file.content_type)
        if not outcome["extracted_values"]:
            raise HTTPException(
                status_code=422,
                detail=(
                    "We couldn't confidently find Hemoglobin, WBC, Platelets, or Glucose "
                    "values in this document. Try a clearer photo, or enter the values manually."
                ),
            )
        return BloodDocumentAnalysisResponse(**outcome)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Blood document parsing failed")
        raise HTTPException(status_code=500, detail=f"Blood document analysis failed: {exc}") from exc


@app.post("/api/analyze-prescription", response_model=PrescriptionAnalysisResponse)
async def analyze_prescription(
    file: UploadFile = File(..., description="Photo or PDF of a prescription"),
) -> PrescriptionAnalysisResponse:
    """Module 4: OCR an uploaded prescription and explain identified medications in plain English."""
    if file.content_type not in ALLOWED_DOCUMENT_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a PNG, JPG, or PDF.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds the 15 MB size limit.")

    try:
        outcome = ml_engine.analyze_prescription_document(file_bytes, file.content_type)
        return PrescriptionAnalysisResponse(**outcome)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prescription analysis failed")
        raise HTTPException(status_code=500, detail=f"Prescription analysis failed: {exc}") from exc


@app.post("/api/symptom-check", response_model=SymptomCheckResponse)
def symptom_check(payload: SymptomCheckRequest) -> SymptomCheckResponse:
    """Module 5: rule-based symptom reasoner — general causes, self-care, and red flags."""
    try:
        outcome = ml_engine.reason_symptom(payload.symptom)
        return SymptomCheckResponse(**outcome)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Symptom check failed")
        raise HTTPException(status_code=500, detail=f"Symptom check failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/{full_path:path}")
def serve_fallback(full_path: str) -> FileResponse:
    """Fallback route so refreshing on a client-side tab route doesn't 404."""
    candidate = STATIC_DIR / full_path
    if candidate.is_file():
        return FileResponse(str(candidate))
    return FileResponse(str(STATIC_DIR / "index.html"))
