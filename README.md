# MedExplain
### Multi-Modal Clinical Intelligence System with Neural Text Simplification and Visual Pathological Grounding

A 3rd-year Computer Science Data Science mini-project demonstrating multiple ML
paradigms in one application — **NLP (Transformers)**, **Computer Vision + XAI
(Grad-CAM)**, **OCR + local knowledge-base reasoning**, and **tabular
rule-based classification** — all running **100% locally**, with **no paid
cloud APIs**.

> ⚠️ **Educational demo only.** This is a coursework project, not a certified
> medical device. All outputs are illustrative and must never be used for
> real clinical decisions.

---

## 1. Architecture Overview

| Module | Technique | Library |
|---|---|---|
| Module 1 — Text Simplification | `google/flan-t5-base` text2text pipeline | 🤗 `transformers` |
| Module 2 — Visual Grounding | `DenseNet-121` (ImageNet weights) + Grad-CAM on `features.denseblock4` | `torch` / `torchvision` / `opencv-python-headless` |
| Module 3 — Biomarker Risk Rules | Reference-range rule engine + weighted risk index | `numpy` / `pydantic` |
| Module 4 — Prescription Analysis | OCR + local curated drug knowledge base + FLAN-T5 summary | `easyocr` / `pymupdf` |
| Module 5 — Symptom Reasoner | Rule-based local knowledge base (causes / self-care / red flags) | pure Python |
| Module 6 — Blood Report Document Parsing | OCR + regex extraction of lab values, feeding Module 3 | `easyocr` / `pymupdf` |

```
MedExplain/
├── requirements.txt
├── ml_engine.py          # Model loading + all 6 modules' core logic
├── main.py               # FastAPI app + routes
├── README.md
└── static/
    ├── index.html         # Sidebar-nav UI: X-Ray, Blood, Prescription, Symptom tabs
    ├── styles.css         # Teal/slate medical theme, sidebar layout
    ├── app.js             # Drag-and-drop, sample cases, fetch calls
    └── samples/
        ├── sample_normal.png          # Synthetic placeholder X-ray
        ├── sample_pneumonia.png       # Synthetic placeholder X-ray
        └── sample_prescription.png    # Synthetic placeholder prescription
```

**Note on sample files:** all three files in `static/samples/` are
procedurally generated synthetic placeholders (drawn with PIL) — no real
patient data is used or required. Swap in real de-identified files any time
by replacing those filenames, or updating the `data-sample` attributes in
`static/index.html`.

---

## 2. What's New (Prescription Analysis, Symptom Reasoner, Document Upload)

- **Prescription Analysis (Module 4):** Upload a photo or PDF of a
  prescription. EasyOCR reads the text, a local curated drug reference (~25
  common medications and their general drug class/purpose) matches any
  recognized medication names, and FLAN-T5 produces a short plain-English
  summary. It **never suggests dosage changes** — only explains what a
  recognized medication is generally used for, and always defers to the
  prescribing doctor/pharmacist.
- **Symptom Reasoner (Module 5):** Type a symptom (e.g. "headache",
  "fatigue", "chest pain") or tap a quick-pick chip. A local, curated
  knowledge base (not LLM-generated, to avoid hallucinated medical claims)
  returns commonly cited general contributing factors, general self-care
  tips, and clear red-flag warning signs that mean "seek care now."
- **Document/photo upload for Blood Report Checker (Module 6):** Instead of
  typing values by hand, upload a photo or PDF of a lab report. OCR +
  regex heuristics extract Hemoglobin/WBC/Platelets/Glucose, pre-fill the
  manual form for verification, and run them through the same rule engine
  as Module 3. Extraction is heuristic — the UI reminds the user to
  double-check extracted numbers.
- **Redesigned UI:** a sidebar-navigation layout (mobile-responsive with a
  hamburger toggle), refreshed typography (Plus Jakarta Sans + Inter via
  Google Fonts), sub-tabs for manual vs. document entry, symptom chips, and
  medication cards.

---

## 3. Prerequisites

- Python 3.10 – 3.12
- ~4 GB free disk space (first run downloads FLAN-T5-base ~990MB, DenseNet-121
  ImageNet weights ~30MB, and EasyOCR's detection+recognition weights
  ~100MB combined — all free, open-source checkpoints, not paid APIs)
- Works on CPU-only machines. Automatically uses CUDA if a GPU is available.
- **No external OCR binary required** — EasyOCR is pure-Python/pip-installable
  (unlike Tesseract, which needs a separate system install).

---

## 4. Step-by-Step Setup (VS Code / Terminal)

### Step 1 — Open the project folder
Open the `MedExplain/` folder in VS Code, or `cd` into it from a terminal.
Prefer a path with no spaces (e.g. `C:\projects\MedExplain`).

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```
In VS Code, select this `venv` as your Python interpreter (`Ctrl+Shift+P` →
"Python: Select Interpreter").

### Step 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This step now also installs `easyocr` and `pymupdf`, so it will take a bit
longer than before (EasyOCR pulls in `scikit-image`, `Shapely`, etc.).

### Step 4 — Run the FastAPI server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The **first** request that touches a given model (text simplification,
X-ray analysis, prescription/blood-document OCR) triggers a one-time
download of that model's weights. This needs internet access *once*; after
that, everything runs fully offline from the local cache (typically
`~/.cache/huggingface`, `~/.cache/torch`, and `~/.EasyOCR`).

### Step 5 — Open the app
Navigate to **http://localhost:8000**. You'll see four sections in the
sidebar:
1. **X-Ray Simplifier** — Grad-CAM heatmap + plain-English explanation.
2. **Blood Report Checker** — manual entry *or* upload a report photo/PDF.
3. **Prescription Analysis** — upload a prescription photo/PDF for OCR +
   medication lookup + plain-English summary.
4. **Symptom Reasoner** — type or tap a symptom to see general causes,
   self-care tips, and red-flag warning signs.

### Step 6 — Interactive API docs (optional)
```
http://localhost:8000/docs
```

---

## 5. Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: sentencepiece` | Run `pip install sentencepiece` (already pinned in requirements.txt) |
| EasyOCR install is slow / large | Normal — it pulls in `scikit-image`/`Shapely`/etc. Let it finish. |
| First OCR/model request hangs for a while | Normal — it's downloading weights. Watch the terminal logs. |
| `CUDA out of memory` | Force CPU by setting `CUDA_VISIBLE_DEVICES=""` before running uvicorn. |
| Port 8000 already in use | `uvicorn main:app --reload --port 8001` |
| Prescription/blood OCR finds nothing | Try a clearer, well-lit photo or higher-resolution scan; PDF text-layer extraction is tried first, then OCR fallback. |
| Blood document extraction pre-fills wrong numbers | Extraction is heuristic regex-based — always verify against the original report before trusting results. |

---

## 6. Grading / Demo Notes

- All modules are exercised end-to-end by the sample buttons — no personal
  data upload is required to demo the project.
- `ml_engine.py` is fully commented and documents the Grad-CAM math
  (Selvaraju et al., 2017), the biomarker reference ranges, the drug
  knowledge base, and the symptom knowledge base.
- No API keys, `.env` files, or paid services are required anywhere in this
  project.
- The Symptom Reasoner and Drug Knowledge Base are intentionally **rule-based
  local lookups**, not LLM-generated text — this avoids hallucinated medical
  claims and keeps every fact traceable to a specific line in `ml_engine.py`.
