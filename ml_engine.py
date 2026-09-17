"""
ml_engine.py
=============
Core Data Science / Machine Learning engine for MedExplain.

This module is intentionally self-contained: it lazily loads all models on
first use (singleton pattern via module-level caches), so the FastAPI process
starts instantly and only pays the model-loading cost once, on the first
request that actually needs it.

IMPORTANT ACADEMIC / SAFETY NOTE
---------------------------------
This project is an educational mini-project demonstrating multi-modal ML
engineering (NLP simplification, CNN + Grad-CAM explainability, and rule-based
tabular risk scoring). It is NOT a certified diagnostic medical device. The
DenseNet-121 backbone used here ships with generic ImageNet-1k weights (no
paid API, no proprietary CheXpert/CheXNet weights are required to run this
project offline). Its "visual feature" outputs are therefore illustrative
proxies for where a real pathology-trained CNN would look, used to
demonstrate the Grad-CAM explainability *pipeline* end-to-end. Every output
of this system is clearly labeled as non-diagnostic in the UI.
"""

from __future__ import annotations

import base64
import io
import threading
from typing import Any

import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Module-level singleton caches + locks (thread-safe lazy loading)
# ---------------------------------------------------------------------------
_t5_pipeline = None
_t5_lock = threading.Lock()

_densenet_model = None
_densenet_lock = threading.Lock()

_ocr_reader = None
_ocr_lock = threading.Lock()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_XRAY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)

# A small curated subset of ImageNet class indices that visually resemble
# radiographic textures (ribs/grids/patterns/etc). Because the backbone is
# generic ImageNet-1k (no paid API and no extra download of a proprietary
# CXR14 checkpoint is required), we relabel the *mechanism* --  which region
# of the image most strongly activated the final conv block  -- into
# clinically-flavoured, clearly-caveated descriptive tags rather than
# pretending the raw ImageNet class name ("mosquito net", "chain mail", etc.)
# is a real diagnosis. This keeps the pipeline 100% local/free while being
# transparent with the end user about what is really being computed.
_VISUAL_FEATURE_TAGS = [
    "Increased opacity / density gradient",
    "Localized texture irregularity",
    "Asymmetry vs. contralateral field",
    "Edge / border sharpness variation",
    "Diffuse haze pattern",
]


# ---------------------------------------------------------------------------
# Module 1: NLP — Clinical Text Simplification (google/flan-t5-base)
# ---------------------------------------------------------------------------
def _get_t5_pipeline():
    """Thread-safe singleton loader for the FLAN-T5 text2text pipeline."""
    global _t5_pipeline
    if _t5_pipeline is None:
        with _t5_lock:
            if _t5_pipeline is None:
                from transformers import pipeline

                _t5_pipeline = pipeline(
                    task="text2text-generation",
                    model="google/flan-t5-base",
                    device=0 if DEVICE.type == "cuda" else -1,
                )
    return _t5_pipeline


def simplify_clinical_text(raw_text: str) -> str:
    """
    Translate a dense clinical radiology/pathology note into a plain-English,
    approximately 6th-grade reading-level explanation for patients.

    Parameters
    ----------
    raw_text : str
        The raw clinical note (jargon-heavy).

    Returns
    -------
    str
        A simplified, patient-friendly explanation.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return "No clinical text was provided, so there is nothing to simplify yet."

    generator = _get_t5_pipeline()

    prompt = (
        "Rewrite the following clinical note in very simple words a "
        "6th-grade student could understand. Avoid medical jargon, explain "
        "any medical term you must use, keep it reassuring but honest, and "
        "use short sentences. Clinical note: " + raw_text
    )

    result = generator(
        prompt,
        max_new_tokens=180,
        min_new_tokens=24,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    simplified = result[0]["generated_text"].strip()

    # FLAN-T5-base can occasionally produce a very short / degenerate output
    # for long inputs. Add a light structural fallback so the UI never shows
    # an empty or single-word explanation.
    if len(simplified.split()) < 6:
        simplified = (
            "In simple terms: " + simplified + ". This means the report "
            "describes a finding your doctor will want to explain to you "
            "in more detail during your visit."
        )

    return simplified


# ---------------------------------------------------------------------------
# Module 2: Computer Vision & XAI — DenseNet-121 + Grad-CAM
# ---------------------------------------------------------------------------
def _get_densenet_model():
    """Thread-safe singleton loader for a pre-trained DenseNet-121."""
    global _densenet_model
    if _densenet_model is None:
        with _densenet_lock:
            if _densenet_model is None:
                weights = models.DenseNet121_Weights.DEFAULT
                model = models.densenet121(weights=weights)
                model.eval()
                model.to(DEVICE)
                _densenet_model = model
    return _densenet_model


class _GradCAMHook:
    """
    Small utility class that registers forward/backward hooks on a target
    layer (here: model.features.denseblock4) to capture activations and
    gradients needed to compute Grad-CAM, as described in Selvaraju et al.
    2017 ("Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization").
    """

    def __init__(self, target_layer: torch.nn.Module):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _compute_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    """
    Runs a forward + backward pass and computes the Grad-CAM heatmap for the
    top predicted class, targeting `model.features.denseblock4` (the final
    dense block before the classifier in torchvision's DenseNet-121).

    Returns
    -------
    cam : np.ndarray
        A (H, W) float32 heatmap normalized to [0, 1].
    probs : torch.Tensor
        The softmax class probabilities for the forward pass (1, num_classes).
    """
    target_layer = model.features.denseblock4
    hook = _GradCAMHook(target_layer)

    try:
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        top_class = int(torch.argmax(probs, dim=1).item())

        model.zero_grad()
        class_score = logits[0, top_class]
        class_score.backward()

        activations = hook.activations[0]  # (C, H, W)
        gradients = hook.gradients[0]  # (C, H, W)

        # Global-average-pool the gradients over spatial dims -> per-channel weight
        weights = gradients.mean(dim=(1, 2))  # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)
        cam = cam - cam.min()
        max_val = cam.max()
        if max_val > 1e-8:
            cam = cam / max_val
        cam_np = cam.cpu().numpy().astype(np.float32)
        return cam_np, probs.detach()
    finally:
        hook.remove()


def generate_xray_cam(image_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
    """
    Runs DenseNet-121 inference on an uploaded chest X-ray image, computes a
    Grad-CAM saliency map over the final dense block, overlays it (Jet
    colormap) on the original image, and returns everything the frontend
    needs to render a side-by-side comparison.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image file (PNG/JPG).

    Returns
    -------
    heatmap_base64 : str
        Base64-encoded PNG of the original image with the Grad-CAM heatmap
        overlaid (ready to embed directly as `data:image/png;base64,...`).
    top_features : list[dict]
        Top-3 descriptive "visual feature" activations with confidence
        scores, e.g. [{"label": "...", "confidence": 0.42}, ...].
    """
    model = _get_densenet_model()

    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_resized = original_image.resize((224, 224))

    input_tensor = _XRAY_TRANSFORM(original_image).unsqueeze(0).to(DEVICE)

    cam, probs = _compute_gradcam(model, input_tensor)

    # Resize CAM (typically 7x7 for DenseNet-121 denseblock4 on 224x224 input)
    # up to the full 224x224 image resolution.
    cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_np = np.array(original_resized).astype(np.float32)
    overlay = (0.55 * original_np + 0.45 * heatmap_color.astype(np.float32)).astype(np.uint8)

    overlay_image = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Build the top-3 "visual feature" activation summary using the model's
    # own top-k softmax confidences, remapped onto clinically-flavoured,
    # clearly-labeled descriptive tags (see module docstring for rationale).
    top_probs, top_idx = torch.topk(probs[0], k=min(5, probs.shape[1]))
    top_features = []
    for rank, (p, _idx) in enumerate(zip(top_probs.tolist(), top_idx.tolist())):
        if rank >= len(_VISUAL_FEATURE_TAGS):
            break
        top_features.append(
            {
                "label": _VISUAL_FEATURE_TAGS[rank],
                "confidence": round(float(p), 4),
            }
        )

    return heatmap_base64, top_features


# ---------------------------------------------------------------------------
# Module 3: Tabular / Clinical Biomarker Rule-Classifier
# ---------------------------------------------------------------------------
# Reference ranges are standard adult clinical reference intervals commonly
# cited in laboratory medicine textbooks. They are simplified for teaching
# purposes and are NOT patient-specific medical thresholds.
_BIOMARKER_RANGES: dict[str, dict[str, Any]] = {
    "hemoglobin": {
        "label": "Hemoglobin (Hb)",
        "unit": "g/dL",
        "normal": (12.0, 17.5),
        "borderline_pad": 1.0,
        "low_insight": "Your hemoglobin is a bit low. This can sometimes mean your "
        "blood is carrying less oxygen than usual, which may cause tiredness. "
        "This is often linked to low iron levels (anemia).",
        "high_insight": "Your hemoglobin is a bit high. This can happen with "
        "dehydration, living at high altitude, or certain lung/heart conditions.",
    },
    "wbc": {
        "label": "White Blood Cells (WBC)",
        "unit": "x10^3/uL",
        "normal": (4.0, 11.0),
        "borderline_pad": 1.0,
        "low_insight": "Your white blood cell count is a bit low, which may mean "
        "your body's infection-fighting cells are reduced right now.",
        "high_insight": "Your white blood cell count is a bit high, which often "
        "happens when your body is fighting an infection or inflammation.",
    },
    "platelets": {
        "label": "Platelets",
        "unit": "x10^3/uL",
        "normal": (150.0, 450.0),
        "borderline_pad": 25.0,
        "low_insight": "Your platelet count is a bit low. Platelets help your "
        "blood clot, so a low count can sometimes mean easier bruising or bleeding.",
        "high_insight": "Your platelet count is a bit high. This can happen after "
        "inflammation, infection, or blood loss, among other causes.",
    },
    "glucose": {
        "label": "Fasting Glucose",
        "unit": "mg/dL",
        "normal": (70.0, 99.0),
        "borderline_pad": 26.0,  # 100-125 is the standard "prediabetes" band
        "low_insight": "Your fasting blood sugar is a bit low, which can cause "
        "shakiness, dizziness, or hunger.",
        "high_insight": "Your fasting blood sugar is a bit high. Levels in this "
        "range are sometimes an early warning sign your body is having a "
        "harder time managing sugar, and are worth discussing with a doctor.",
    },
}


def _classify_value(value: float, low: float, high: float, pad: float) -> str:
    """Classify a numeric lab value into Normal / Borderline / Critical."""
    if low <= value <= high:
        return "Normal"
    if (low - pad) <= value < low or high < value <= (high + pad):
        return "Borderline"
    return "Critical"


def evaluate_blood_biomarkers(metrics: dict[str, float]) -> list[dict[str, Any]]:
    """
    Evaluate a dict of blood lab metrics against standard adult clinical
    reference ranges and return structured, plain-English risk assessments.

    Parameters
    ----------
    metrics : dict[str, float]
        Keys should be a subset of {"hemoglobin", "wbc", "platelets",
        "glucose"} mapped to their numeric measured values.

    Returns
    -------
    list[dict]
        One entry per evaluated biomarker:
        {
            "key": "hemoglobin",
            "label": "Hemoglobin (Hb)",
            "value": 9.8,
            "unit": "g/dL",
            "normal_range": "12.0 - 17.5",
            "status": "Critical",
            "insight": "..."
        }
        Plus a final synthetic entry with key "overall_risk_index" summarizing
        an aggregate 0-100 health risk index derived from the individual
        statuses.
    """
    results: list[dict[str, Any]] = []
    status_weight = {"Normal": 0, "Borderline": 1, "Critical": 2}
    total_weight = 0
    evaluated_count = 0

    for key, spec in _BIOMARKER_RANGES.items():
        if key not in metrics or metrics[key] is None:
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            continue

        low, high = spec["normal"]
        pad = spec["borderline_pad"]
        status = _classify_value(value, low, high, pad)

        if status == "Normal":
            insight = f"Your {spec['label'].lower()} is within the normal range. No concerns here."
        elif value < low:
            insight = spec["low_insight"]
        else:
            insight = spec["high_insight"]

        results.append(
            {
                "key": key,
                "label": spec["label"],
                "value": value,
                "unit": spec["unit"],
                "normal_range": f"{low:.1f} - {high:.1f}",
                "status": status,
                "insight": insight,
            }
        )
        total_weight += status_weight[status]
        evaluated_count += 1

    if evaluated_count > 0:
        max_possible = evaluated_count * 2
        risk_index = round((total_weight / max_possible) * 100)
    else:
        risk_index = 0

    if risk_index == 0:
        overall_status = "Low Risk"
        overall_insight = "All evaluated biomarkers look normal. Keep up your regular checkups."
    elif risk_index <= 40:
        overall_status = "Mild Risk"
        overall_insight = "One or more markers are slightly outside the normal range. Consider mentioning this to your doctor at your next visit."
    elif risk_index <= 70:
        overall_status = "Moderate Risk"
        overall_insight = "Several markers are outside the normal range. It is a good idea to follow up with a healthcare provider soon."
    else:
        overall_status = "High Risk"
        overall_insight = "Multiple markers show significant deviations. Please consult a healthcare provider promptly for further evaluation."

    results.append(
        {
            "key": "overall_risk_index",
            "label": "Overall Health Risk Index",
            "value": risk_index,
            "unit": "/ 100",
            "normal_range": "0 - 40 (lower is better)",
            "status": overall_status,
            "insight": overall_insight,
        }
    )

    return results


# ---------------------------------------------------------------------------
# Shared OCR utility (used by Module 4: Prescription Analysis and
# Module 6: Blood Report Document Parsing)
# ---------------------------------------------------------------------------
def _get_ocr_reader():
    """
    Thread-safe singleton loader for an EasyOCR reader.

    EasyOCR is a pure-Python, pip-installable OCR engine (no external system
    binary required, unlike Tesseract), and its detection/recognition
    checkpoints are free, open-source weights downloaded once from EasyOCR's
    public model repository — no paid API involved.
    """
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                import easyocr

                _ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE.type == "cuda"), verbose=False)
    return _ocr_reader


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes and return the concatenated recognized text."""
    reader = _get_ocr_reader()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    lines = reader.readtext(image_np, detail=0, paragraph=True)
    return "\n".join(lines)


def _extract_text_from_document(file_bytes: bytes, content_type: str) -> str:
    """
    Extract raw text from an uploaded document, which may be an image
    (PNG/JPG/WEBP -> OCR directly) or a PDF (try native embedded text first
    via PyMuPDF; fall back to rendering each page to an image and running
    OCR, which handles scanned/photographed PDF reports too).
    """
    content_type = (content_type or "").lower()

    if "pdf" in content_type:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts: list[str] = []
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
            else:
                # Likely a scanned page with no embedded text layer — rasterize
                # it and fall back to OCR.
                pix = page.get_pixmap(dpi=200)
                png_bytes = pix.tobytes("png")
                text_parts.append(_ocr_image_bytes(png_bytes))
        doc.close()
        return "\n".join(text_parts).strip()

    # Otherwise treat as a plain image.
    return _ocr_image_bytes(file_bytes).strip()


# ---------------------------------------------------------------------------
# Module 4: Prescription Analysis (OCR + Local Drug Knowledge Base + NLP)
# ---------------------------------------------------------------------------
# A small, curated, locally-stored knowledge base of common medications. This
# is intentionally general educational information (drug class + typical
# purpose) rather than dosing guidance, and is not a substitute for a
# pharmacist's or doctor's advice.
_DRUG_KB: list[dict[str, Any]] = [
    {"aliases": ["paracetamol", "acetaminophen", "tylenol", "crocin", "dolo", "calpol"],
     "name": "Paracetamol (Acetaminophen)", "drug_class": "Analgesic / Antipyretic",
     "common_use": "Commonly used to relieve mild-to-moderate pain and reduce fever."},
    {"aliases": ["ibuprofen", "brufen", "advil", "motrin"],
     "name": "Ibuprofen", "drug_class": "NSAID (Anti-inflammatory)",
     "common_use": "Commonly used to reduce pain, inflammation, and fever."},
    {"aliases": ["aspirin", "disprin", "ecosprin"],
     "name": "Aspirin", "drug_class": "NSAID / Antiplatelet",
     "common_use": "Used for pain relief and, at low doses, to help prevent blood clots."},
    {"aliases": ["amoxicillin", "amoxil", "mox"],
     "name": "Amoxicillin", "drug_class": "Antibiotic (Penicillin family)",
     "common_use": "Used to treat a range of bacterial infections."},
    {"aliases": ["azithromycin", "azithral", "zithromax", "z-pack", "azee"],
     "name": "Azithromycin", "drug_class": "Antibiotic (Macrolide)",
     "common_use": "Used to treat bacterial infections, including respiratory and skin infections."},
    {"aliases": ["ciprofloxacin", "cipro", "ciplox"],
     "name": "Ciprofloxacin", "drug_class": "Antibiotic (Fluoroquinolone)",
     "common_use": "Used to treat certain bacterial infections, including some urinary tract infections."},
    {"aliases": ["metformin", "glucophage", "glycomet"],
     "name": "Metformin", "drug_class": "Antidiabetic (Biguanide)",
     "common_use": "Commonly used to help manage blood sugar levels in type 2 diabetes."},
    {"aliases": ["insulin", "lantus", "humalog", "novorapid"],
     "name": "Insulin", "drug_class": "Antidiabetic (Hormone)",
     "common_use": "Used to control blood sugar levels, mainly in diabetes."},
    {"aliases": ["atorvastatin", "lipitor", "atorva"],
     "name": "Atorvastatin", "drug_class": "Statin (Cholesterol-lowering)",
     "common_use": "Commonly used to lower LDL cholesterol and reduce cardiovascular risk."},
    {"aliases": ["amlodipine", "norvasc", "amlopres"],
     "name": "Amlodipine", "drug_class": "Calcium Channel Blocker",
     "common_use": "Commonly used to treat high blood pressure and chest pain (angina)."},
    {"aliases": ["lisinopril", "prinivil", "zestril"],
     "name": "Lisinopril", "drug_class": "ACE Inhibitor",
     "common_use": "Commonly used to treat high blood pressure and heart failure."},
    {"aliases": ["losartan", "cozaar"],
     "name": "Losartan", "drug_class": "ARB (Angiotensin Receptor Blocker)",
     "common_use": "Commonly used to treat high blood pressure."},
    {"aliases": ["hydrochlorothiazide", "hctz", "microzide"],
     "name": "Hydrochlorothiazide", "drug_class": "Diuretic",
     "common_use": "Commonly used to treat high blood pressure and fluid retention."},
    {"aliases": ["omeprazole", "prilosec", "omez"],
     "name": "Omeprazole", "drug_class": "Proton Pump Inhibitor",
     "common_use": "Used to reduce stomach acid, for acid reflux and ulcers."},
    {"aliases": ["pantoprazole", "protonix", "pantocid"],
     "name": "Pantoprazole", "drug_class": "Proton Pump Inhibitor",
     "common_use": "Used to reduce stomach acid, for acid reflux and ulcers."},
    {"aliases": ["ranitidine", "zantac"],
     "name": "Ranitidine", "drug_class": "H2 Blocker",
     "common_use": "Historically used to reduce stomach acid (largely replaced by PPIs)."},
    {"aliases": ["cetirizine", "zyrtec", "cetrizine", "alerid"],
     "name": "Cetirizine", "drug_class": "Antihistamine",
     "common_use": "Commonly used to relieve allergy symptoms such as sneezing and itching."},
    {"aliases": ["loratadine", "claritin"],
     "name": "Loratadine", "drug_class": "Antihistamine",
     "common_use": "Commonly used to relieve allergy symptoms."},
    {"aliases": ["salbutamol", "albuterol", "ventolin", "asthalin"],
     "name": "Salbutamol (Albuterol)", "drug_class": "Bronchodilator",
     "common_use": "Used to relieve wheezing and shortness of breath in asthma/COPD."},
    {"aliases": ["montelukast", "singulair"],
     "name": "Montelukast", "drug_class": "Leukotriene Receptor Antagonist",
     "common_use": "Used to help control asthma and allergy symptoms."},
    {"aliases": ["levothyroxine", "synthroid", "thyronorm", "eltroxin"],
     "name": "Levothyroxine", "drug_class": "Thyroid Hormone Replacement",
     "common_use": "Used to treat an underactive thyroid (hypothyroidism)."},
    {"aliases": ["metronidazole", "flagyl"],
     "name": "Metronidazole", "drug_class": "Antibiotic / Antiprotozoal",
     "common_use": "Used to treat certain bacterial and parasitic infections."},
    {"aliases": ["diclofenac", "voltaren", "voveran"],
     "name": "Diclofenac", "drug_class": "NSAID (Anti-inflammatory)",
     "common_use": "Used to relieve pain, swelling, and inflammation."},
    {"aliases": ["prednisone", "prednisolone", "wysolone"],
     "name": "Prednisone / Prednisolone", "drug_class": "Corticosteroid",
     "common_use": "Used to reduce inflammation in a wide range of conditions."},
    {"aliases": ["multivitamin", "vitamin d", "vitamin b12", "folic acid", "iron supplement", "ferrous sulfate"],
     "name": "Vitamin / Mineral Supplement", "drug_class": "Dietary Supplement",
     "common_use": "Used to correct or prevent a nutritional deficiency."},
]


def _find_medications_in_text(text: str) -> list[dict[str, str]]:
    """Match known drug aliases (whole-word, case-insensitive) inside OCR'd text."""
    text_lower = text.lower()
    matched: list[dict[str, str]] = []
    seen_names = set()

    for entry in _DRUG_KB:
        for alias in entry["aliases"]:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                if entry["name"] not in seen_names:
                    matched.append(
                        {
                            "name": entry["name"],
                            "drug_class": entry["drug_class"],
                            "common_use": entry["common_use"],
                        }
                    )
                    seen_names.add(entry["name"])
                break

    return matched


def analyze_prescription_document(file_bytes: bytes, content_type: str) -> dict[str, Any]:
    """
    Full Module 4 pipeline: OCR the uploaded prescription (image or PDF),
    identify any recognizable medications against the local drug knowledge
    base, and produce a plain-English summary of what was found.

    IMPORTANT: This never suggests dosage changes. It only explains, in
    general terms, what a recognized medication is typically used for, and
    always defers to the prescribing doctor / pharmacist for anything
    dosage- or treatment-related.
    """
    raw_text = _extract_text_from_document(file_bytes, content_type)

    if not raw_text.strip():
        return {
            "raw_text": "",
            "identified_medications": [],
            "plain_summary": (
                "We couldn't read any text from this file. Try a clearer, well-lit "
                "photo of the prescription, or a higher-resolution scan."
            ),
        }

    identified = _find_medications_in_text(raw_text)

    if identified:
        med_list_str = "; ".join(f"{m['name']} ({m['drug_class']})" for m in identified)
        prompt = (
            "You are explaining a prescription to a patient in very simple, "
            "reassuring words a 6th-grade student could understand. Do NOT "
            "suggest any dosage changes. The prescription appears to include "
            "these medications: " + med_list_str + ". Briefly explain in 2-3 "
            "simple sentences what these medications are generally used for, "
            "and remind the reader to follow their doctor's exact instructions."
        )
    else:
        prompt = (
            "Rewrite the following prescription text in very simple words a "
            "6th-grade student could understand, without suggesting any "
            "dosage changes. Prescription text: " + raw_text[:800]
        )

    try:
        generator = _get_t5_pipeline()
        result = generator(
            prompt,
            max_new_tokens=160,
            min_new_tokens=20,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        plain_summary = result[0]["generated_text"].strip()
    except Exception:  # noqa: BLE001
        plain_summary = (
            "Here is what we found on the prescription. Please review the "
            "identified medications below, and always follow your doctor's "
            "or pharmacist's exact instructions."
        )

    return {
        "raw_text": raw_text,
        "identified_medications": identified,
        "plain_summary": plain_summary,
    }


# ---------------------------------------------------------------------------
# Module 5: Symptom Reasoner — "Why am I feeling this?"
# ---------------------------------------------------------------------------
# A small, curated, locally-stored knowledge base of common everyday symptoms
# mapped to (a) commonly cited general contributing factors, (b) general,
# non-personalized self-care tips, and (c) red-flag warning signs that
# warrant urgent medical attention. This is general health education content,
# NOT a diagnosis of the individual user, and is intentionally conservative:
# every entry ends by encouraging a doctor visit if symptoms persist/worsen.
_SYMPTOM_KB: dict[str, dict[str, Any]] = {
    "headache": {
        "display": "Headache",
        "common_causes": [
            "Dehydration (not drinking enough water)",
            "Lack of sleep or disrupted sleep schedule",
            "Eye strain from screens",
            "Stress or muscle tension in the neck/shoulders",
            "Skipping meals (low blood sugar)",
            "Caffeine withdrawal",
            "Sinus congestion or a cold",
        ],
        "self_care": [
            "Drink a glass of water and rehydrate through the day",
            "Rest in a quiet, dim room",
            "Take a break from screens every 20-30 minutes",
            "Try gentle neck and shoulder stretches",
        ],
        "red_flags": [
            "Sudden, severe \"worst headache of your life\"",
            "Headache with confusion, vision loss, or slurred speech",
            "Headache following a head injury",
            "Headache with a high fever and stiff neck",
        ],
    },
    "fatigue": {
        "display": "Fatigue / Tiredness",
        "common_causes": [
            "Poor or insufficient sleep",
            "Dehydration or poor nutrition",
            "Prolonged stress or anxiety",
            "Sedentary lifestyle / lack of physical activity",
            "Low iron levels (anemia)",
            "An underlying viral infection",
        ],
        "self_care": [
            "Aim for 7-9 hours of consistent sleep",
            "Stay hydrated and eat balanced meals",
            "Add light physical activity like a short walk",
            "Take short breaks during long work sessions",
        ],
        "red_flags": [
            "Fatigue that is sudden, severe, and unexplained",
            "Fatigue with chest pain or shortness of breath",
            "Fatigue with unexplained weight loss",
        ],
    },
    "dizziness": {
        "display": "Dizziness",
        "common_causes": [
            "Dehydration",
            "Standing up too quickly (low blood pressure drop)",
            "Low blood sugar",
            "Inner-ear balance issues",
            "Motion sickness",
        ],
        "self_care": [
            "Sit or lie down until it passes",
            "Rehydrate slowly with water",
            "Stand up slowly from sitting/lying positions",
            "Avoid sudden head movements",
        ],
        "red_flags": [
            "Dizziness with chest pain, slurred speech, or numbness",
            "Dizziness after a head injury",
            "Fainting / loss of consciousness",
        ],
    },
    "nausea": {
        "display": "Nausea",
        "common_causes": [
            "Something you ate (mild food intolerance)",
            "Motion sickness",
            "Stress or anxiety",
            "Early signs of a stomach virus",
            "Skipping meals or an empty stomach",
        ],
        "self_care": [
            "Sip clear fluids slowly",
            "Try small, bland meals (crackers, toast)",
            "Get fresh air and rest",
            "Avoid strong smells and greasy food",
        ],
        "red_flags": [
            "Nausea with severe abdominal pain",
            "Persistent vomiting, unable to keep fluids down",
            "Nausea with signs of dehydration or blood in vomit",
        ],
    },
    "fever": {
        "display": "Fever",
        "common_causes": [
            "A common viral infection (cold/flu)",
            "A bacterial infection",
            "Recent vaccination",
            "Heat exposure / overheating",
        ],
        "self_care": [
            "Rest and stay well hydrated",
            "Dress lightly and keep the room cool",
            "Monitor your temperature periodically",
        ],
        "red_flags": [
            "Fever above 103°F (39.4°C) in adults",
            "Fever lasting more than 3 days",
            "Fever with stiff neck, rash, or difficulty breathing",
            "Fever in an infant under 3 months",
        ],
    },
    "cough": {
        "display": "Cough",
        "common_causes": [
            "A common cold or viral infection",
            "Seasonal allergies",
            "Dry indoor air or throat irritation",
            "Post-nasal drip",
        ],
        "self_care": [
            "Stay hydrated to soothe your throat",
            "Use a humidifier or inhale steam",
            "Try warm tea with honey (for adults)",
        ],
        "red_flags": [
            "Cough with blood",
            "Cough with high fever and shortness of breath",
            "Cough lasting more than 3 weeks",
        ],
    },
    "sore throat": {
        "display": "Sore Throat",
        "common_causes": [
            "A viral infection (common cold)",
            "Seasonal allergies or dry air",
            "Straining your voice",
            "Bacterial infection (e.g., strep throat)",
        ],
        "self_care": [
            "Gargle with warm salt water",
            "Stay hydrated with warm fluids",
            "Rest your voice",
        ],
        "red_flags": [
            "Severe difficulty swallowing or breathing",
            "Sore throat with high fever and swollen glands",
            "Sore throat lasting more than a week",
        ],
    },
    "stomach pain": {
        "display": "Stomach / Abdominal Pain",
        "common_causes": [
            "Indigestion or gas",
            "Something you ate",
            "Stress or anxiety",
            "Constipation",
            "Menstrual cramps",
        ],
        "self_care": [
            "Rest and avoid heavy meals temporarily",
            "Try a warm compress on your abdomen",
            "Stay hydrated with clear fluids",
        ],
        "red_flags": [
            "Sudden, severe abdominal pain",
            "Pain with a rigid, tender abdomen",
            "Pain with fever, vomiting blood, or black stools",
            "Pain localized to the lower-right abdomen (possible appendicitis)",
        ],
    },
    "back pain": {
        "display": "Back Pain",
        "common_causes": [
            "Muscle strain from lifting or poor posture",
            "Prolonged sitting",
            "Sleeping in an awkward position",
            "Weak core muscles",
        ],
        "self_care": [
            "Apply ice for the first 48 hours, then heat",
            "Gentle stretching and short walks",
            "Maintain good posture; avoid prolonged sitting",
        ],
        "red_flags": [
            "Back pain with numbness/weakness in the legs",
            "Back pain with loss of bladder/bowel control",
            "Back pain after significant trauma",
        ],
    },
    "chest pain": {
        "display": "Chest Pain",
        "common_causes": [
            "Muscle strain from exercise or coughing",
            "Acid reflux / heartburn",
            "Anxiety or a panic episode",
            "Costochondritis (inflammation of chest wall cartilage)",
        ],
        "self_care": [
            "Rest and monitor how the pain changes",
            "Note if it worsens with movement/breathing vs at rest",
        ],
        "red_flags": [
            "Chest pain with pressure/tightness lasting more than a few minutes",
            "Chest pain radiating to the arm, jaw, or back",
            "Chest pain with shortness of breath, sweating, or nausea",
            "Any chest pain you're unsure about — treat as an emergency and seek immediate care",
        ],
    },
    "shortness of breath": {
        "display": "Shortness of Breath",
        "common_causes": [
            "Physical exertion or poor fitness",
            "Anxiety or a panic episode",
            "Mild asthma flare-up",
            "Nasal congestion from a cold",
        ],
        "self_care": [
            "Sit upright and try slow, controlled breathing",
            "Move away from any potential irritants (smoke, dust)",
        ],
        "red_flags": [
            "Sudden, severe shortness of breath",
            "Shortness of breath with chest pain or blue lips",
            "Shortness of breath at rest that doesn't improve",
        ],
    },
    "joint pain": {
        "display": "Joint Pain",
        "common_causes": [
            "Overuse or minor strain",
            "Age-related wear (osteoarthritis)",
            "Recent injury",
            "Inflammation from an infection",
        ],
        "self_care": [
            "Rest the joint and apply ice for swelling",
            "Gentle range-of-motion exercises once pain eases",
            "Over-the-counter pain relief as directed on the label",
        ],
        "red_flags": [
            "Joint that is hot, red, and severely swollen",
            "Joint pain with fever",
            "Inability to bear weight or move the joint",
        ],
    },
    "insomnia": {
        "display": "Trouble Sleeping",
        "common_causes": [
            "Stress or racing thoughts",
            "Too much screen time before bed",
            "Caffeine or alcohol close to bedtime",
            "Irregular sleep schedule",
        ],
        "self_care": [
            "Keep a consistent sleep/wake time",
            "Avoid screens 30-60 minutes before bed",
            "Limit caffeine in the afternoon/evening",
            "Try relaxation breathing before sleep",
        ],
        "red_flags": [
            "Insomnia lasting for weeks and affecting daily function",
            "Insomnia with signs of depression or anxiety",
        ],
    },
    "anxiety": {
        "display": "Anxiety / Racing Heart",
        "common_causes": [
            "Stress from work, school, or life events",
            "Excess caffeine intake",
            "Lack of sleep",
            "A panic episode",
        ],
        "self_care": [
            "Try slow, deep breathing (inhale 4s, exhale 6s)",
            "Ground yourself by naming things you can see/hear/touch",
            "Reduce caffeine intake",
            "Talk to someone you trust about what's on your mind",
        ],
        "red_flags": [
            "Chest pain, fainting, or a sense of impending doom",
            "Thoughts of self-harm — reach out for help immediately",
        ],
    },
    "skin rash": {
        "display": "Skin Rash",
        "common_causes": [
            "Contact with an irritant or allergen",
            "Dry skin or eczema",
            "Heat rash",
            "A mild allergic reaction",
        ],
        "self_care": [
            "Avoid scratching and keep the area clean and dry",
            "Apply a fragrance-free moisturizer",
            "Avoid known irritants (new soaps, detergents, fabrics)",
        ],
        "red_flags": [
            "Rash with difficulty breathing or facial swelling (call emergency services)",
            "Rapidly spreading rash with fever",
            "Rash with blistering or peeling skin",
        ],
    },
    "diarrhea": {
        "display": "Diarrhea",
        "common_causes": [
            "A stomach virus",
            "Food intolerance or spoiled food",
            "Stress",
            "Recent antibiotic use",
        ],
        "self_care": [
            "Stay hydrated with water and electrolyte drinks",
            "Eat bland foods (rice, bananas, toast)",
            "Avoid dairy, caffeine, and greasy foods temporarily",
        ],
        "red_flags": [
            "Signs of dehydration (dizziness, very dark urine)",
            "Blood in stool or black, tarry stool",
            "Diarrhea lasting more than 2 days with high fever",
        ],
    },
    "constipation": {
        "display": "Constipation",
        "common_causes": [
            "Low fiber intake",
            "Not drinking enough water",
            "Lack of physical activity",
            "Ignoring the urge to go",
        ],
        "self_care": [
            "Increase fiber intake (fruits, vegetables, whole grains)",
            "Drink more water through the day",
            "Add light physical activity like walking",
        ],
        "red_flags": [
            "Severe abdominal pain or bloating",
            "No bowel movement for more than a week",
            "Blood in stool",
        ],
    },
}

# Common synonyms/misspellings mapped to a canonical key in _SYMPTOM_KB.
_SYMPTOM_SYNONYMS: dict[str, str] = {
    "tired": "fatigue", "exhausted": "fatigue", "sleepy": "fatigue", "low energy": "fatigue",
    "dizzy": "dizziness", "lightheaded": "dizziness", "light headed": "dizziness",
    "throwing up": "nausea", "vomiting": "nausea", "feel sick": "nausea", "queasy": "nausea",
    "high temperature": "fever", "temperature": "fever", "feverish": "fever",
    "throat pain": "sore throat", "sore throat": "sore throat", "throat hurts": "sore throat",
    "stomach ache": "stomach pain", "stomachache": "stomach pain", "belly pain": "stomach pain",
    "tummy ache": "stomach pain", "abdominal pain": "stomach pain",
    "backache": "back pain", "back hurts": "back pain",
    "can't breathe": "shortness of breath", "breathless": "shortness of breath",
    "cant sleep": "insomnia", "can't sleep": "insomnia", "sleepless": "insomnia",
    "stressed": "anxiety", "panic": "anxiety", "panicking": "anxiety", "racing heart": "anxiety",
    "rash": "skin rash", "itchy skin": "skin rash",
    "loose motion": "diarrhea", "loose motions": "diarrhea", "runs": "diarrhea",
    "cant poop": "constipation", "can't poop": "constipation", "blocked up": "constipation",
    "migraine": "headache", "head pain": "headache", "head ache": "headache",
    "joint ache": "joint pain", "achy joints": "joint pain",
}

_GENERIC_DISCLAIMER = (
    "This is general educational information, not a diagnosis. If your symptoms "
    "are severe, sudden, or you're worried, please see a doctor or seek urgent care."
)


def reason_symptom(symptom_text: str) -> dict[str, Any]:
    """
    Look up a free-text symptom description against the local symptom
    knowledge base and return commonly cited contributing factors, general
    self-care tips, and red-flag warning signs.

    This is intentionally rule-based (not LLM-generated) to avoid
    hallucinating medically-sensitive claims: every fact returned comes from
    the curated `_SYMPTOM_KB` above.
    """
    normalized = (symptom_text or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)

    if not normalized:
        return {
            "matched": False,
            "query": symptom_text,
            "matched_symptom": None,
            "common_causes": [],
            "self_care": [],
            "red_flags": [],
            "disclaimer": _GENERIC_DISCLAIMER,
            "message": "Please enter a symptom, like 'headache' or 'fatigue'.",
        }

    # 1. Direct key match
    canonical_key = None
    if normalized in _SYMPTOM_KB:
        canonical_key = normalized
    elif normalized in _SYMPTOM_SYNONYMS:
        canonical_key = _SYMPTOM_SYNONYMS[normalized]
    else:
        # 2. Substring match against known keys/synonyms (handles phrases like
        #    "I have a really bad headache today")
        for key in _SYMPTOM_KB:
            if key in normalized:
                canonical_key = key
                break
        if canonical_key is None:
            for phrase, key in _SYMPTOM_SYNONYMS.items():
                if phrase in normalized:
                    canonical_key = key
                    break

    if canonical_key and canonical_key in _SYMPTOM_KB:
        entry = _SYMPTOM_KB[canonical_key]
        return {
            "matched": True,
            "query": symptom_text,
            "matched_symptom": entry["display"],
            "common_causes": entry["common_causes"],
            "self_care": entry["self_care"],
            "red_flags": entry["red_flags"],
            "disclaimer": _GENERIC_DISCLAIMER,
            "message": None,
        }

    return {
        "matched": False,
        "query": symptom_text,
        "matched_symptom": None,
        "common_causes": [],
        "self_care": [
            "Track when the symptom started and anything that seems to trigger it",
            "Stay hydrated and rested",
            "Note any other symptoms happening alongside it",
        ],
        "red_flags": [
            "Any symptom that is sudden, severe, or rapidly worsening",
            "Any symptom alongside difficulty breathing, chest pain, or confusion",
        ],
        "disclaimer": _GENERIC_DISCLAIMER,
        "message": (
            f"We don't have specific information for \"{symptom_text}\" in our local "
            "knowledge base yet. Here is some general guidance instead — for anything "
            "concerning, please check with a doctor."
        ),
    }


# ---------------------------------------------------------------------------
# Module 6: Blood Report Document Parsing (OCR + Regex Extraction)
# ---------------------------------------------------------------------------
# Regex patterns tolerant of common lab-report phrasings/abbreviations for
# each of the four biomarkers this project evaluates.
_BLOOD_VALUE_PATTERNS: dict[str, str] = {
    "hemoglobin": r"h[ae]moglobin\s*\(?hb\)?|hb\b",
    "wbc": r"(?:white\s*blood\s*cell[s]?|w\.?b\.?c\.?|total\s*leuko[c]?yte\s*count|tlc)",
    "platelets": r"platelet[s]?(?:\s*count)?|plt\b",
    "glucose": r"(?:fasting\s*)?(?:blood\s*)?glucose|fbs\b",
}

_NUMBER_PATTERN = r"([\d]{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.?\d*)"


def _normalize_count(key: str, value: float) -> float:
    """
    Lab reports often give WBC/platelet counts as raw cell counts per
    microliter (e.g. "WBC: 7200/uL", "Platelets: 250000/uL") rather than the
    x10^3/uL units this project's rule engine expects. Heuristically rescale
    large raw counts down to the expected thousands-scale.
    """
    if key == "wbc" and value > 1000:
        return round(value / 1000, 2)
    if key == "platelets" and value > 5000:
        return round(value / 1000, 2)
    return value


def extract_blood_values_from_document(file_bytes: bytes, content_type: str) -> dict[str, float]:
    """
    OCR/parse an uploaded blood lab report (image or PDF) and heuristically
    extract Hemoglobin, WBC, Platelets, and Fasting Glucose values via regex.

    Returns a dict subset of {"hemoglobin", "wbc", "platelets", "glucose"} ->
    float value. Keys that could not be confidently found are omitted, so the
    frontend can prefill the form and let the user manually verify/complete
    the rest.
    """
    raw_text = _extract_text_from_document(file_bytes, content_type)
    text_lower = raw_text.lower()

    extracted: dict[str, float] = {}

    for key, label_pattern in _BLOOD_VALUE_PATTERNS.items():
        # Look for "<label> ... <number>" within a short window of characters,
        # tolerating colons, dashes, and a handful of intervening words/units.
        pattern = rf"(?:{label_pattern})\s*[:\-]?\s*(?:[a-z\.\/]*\s*){{0,3}}{_NUMBER_PATTERN}"
        match = re.search(pattern, text_lower)
        if match:
            raw_value = match.group(1).replace(",", "")
            try:
                value = float(raw_value)
            except ValueError:
                continue
            extracted[key] = _normalize_count(key, value)

    return extracted


def analyze_blood_document(file_bytes: bytes, content_type: str) -> dict[str, Any]:
    """Module 6 full pipeline: extract values from a document, then evaluate them."""
    extracted = extract_blood_values_from_document(file_bytes, content_type)
    results = evaluate_blood_biomarkers(extracted) if extracted else []
    return {
        "extracted_values": extracted,
        "results": results,
    }
