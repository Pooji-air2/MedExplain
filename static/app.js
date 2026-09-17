"use strict";

/* ==========================================================================
   MedExplain — app.js
   Handles: sidebar navigation, drag-and-drop uploads, sample cases,
   asynchronous API calls, and DOM rendering for all 4 modules.
   ========================================================================== */

const API_BASE = "";

/* ---------------------------------------------------------------------- */
/* Toast helper                                                            */
/* ---------------------------------------------------------------------- */
let toastTimeout = null;
function showToast(message, isError = false) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.style.background = isError ? "#991b1b" : "#1a2332";
  toast.classList.remove("hidden");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.add("hidden"), 3200);
}

/* ---------------------------------------------------------------------- */
/* Sidebar navigation + mobile toggle                                      */
/* ---------------------------------------------------------------------- */
function initNav() {
  const navButtons = document.querySelectorAll(".nav-item");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      navButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");

      // Close mobile sidebar after navigation
      document.getElementById("sidebar").classList.remove("open");
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  document.getElementById("mobileNavToggle").addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
  });
}

function initSubTabs() {
  document.querySelectorAll(".sub-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const container = btn.closest(".card");
      container.querySelectorAll(".sub-tab-btn").forEach((b) => b.classList.remove("active"));
      container.querySelectorAll(".sub-tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.subtab).classList.add("active");
    });
  });
}

/* ========================================================================
   TAB 1: X-RAY & FINDINGS SIMPLIFIER
   ======================================================================== */

let currentXrayFile = null;

const dropzone = document.getElementById("dropzone");
const xrayFileInput = document.getElementById("xrayFileInput");
const analyzeXrayBtn = document.getElementById("analyzeXrayBtn");
const clinicalNoteInput = document.getElementById("clinicalNoteInput");
const originalImagePreview = document.getElementById("originalImagePreview");
const xrayStatusBadge = document.getElementById("xrayStatusBadge");
const xrayLoading = document.getElementById("xrayLoading");
const xrayEmptyState = document.getElementById("xrayEmptyState");
const xrayResults = document.getElementById("xrayResults");

function setBadge(el, text, variant) {
  el.textContent = text;
  el.className = "badge " + variant;
}

function handleSelectedXrayFile(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg", "image/jpg", "image/webp"].includes(file.type)) {
    showToast("Please choose a PNG, JPG, or WEBP image.", true);
    return;
  }
  currentXrayFile = file;
  analyzeXrayBtn.disabled = false;
  setBadge(xrayStatusBadge, "Image ready", "badge-warning");

  const reader = new FileReader();
  reader.onload = (e) => { originalImagePreview.src = e.target.result; };
  reader.readAsDataURL(file);

  showToast(`Loaded "${file.name}"`);
}

function initDropzone() {
  dropzone.addEventListener("click", () => xrayFileInput.click());
  xrayFileInput.addEventListener("change", (e) => handleSelectedXrayFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((evtName) => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evtName) => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    handleSelectedXrayFile(file);
  });
}

const SAMPLE_NOTES = {
  normal: "The lungs are clear bilaterally with no focal consolidation, effusion, or pneumothorax. Cardiac silhouette is within normal limits. No acute cardiopulmonary abnormality.",
  pneumonia: "Focal airspace opacity is noted in the right lower lobe with associated air bronchograms, findings consistent with lobar pneumonia. No pleural effusion or pneumothorax identified. Recommend clinical correlation and follow-up imaging after treatment.",
};

async function fetchAsFile(url, filename, mimeType) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error("Could not load sample file");
  const blob = await resp.blob();
  return new File([blob], filename, { type: mimeType });
}

function initSampleXrayButtons() {
  document.querySelectorAll(".sample-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const filename = btn.dataset.sample;
      const noteKey = btn.dataset.note;
      try {
        const file = await fetchAsFile(`/static/samples/${filename}`, filename, "image/png");
        handleSelectedXrayFile(file);
        clinicalNoteInput.value = SAMPLE_NOTES[noteKey] || "";
      } catch (err) {
        showToast("Failed to load sample case.", true);
        console.error(err);
      }
    });
  });
}

function renderFeaturesList(features) {
  const list = document.getElementById("featuresList");
  list.innerHTML = "";
  features.forEach((f) => {
    const pct = Math.round(f.confidence * 100);
    const li = document.createElement("li");
    li.innerHTML = `
      <span style="min-width: 190px;">${f.label}</span>
      <span class="feature-bar-track"><span class="feature-bar-fill" style="width:${pct}%"></span></span>
      <span class="feature-pct">${pct}%</span>
    `;
    list.appendChild(li);
  });
}

async function analyzeXray() {
  if (!currentXrayFile) {
    showToast("Please upload an image first.", true);
    return;
  }

  xrayEmptyState.classList.add("hidden");
  xrayResults.classList.add("hidden");
  xrayLoading.classList.remove("hidden");
  analyzeXrayBtn.disabled = true;
  setBadge(xrayStatusBadge, "Analyzing…", "badge-warning");

  try {
    const formData = new FormData();
    formData.append("image", currentXrayFile);
    formData.append("clinical_text", clinicalNoteInput.value || "");

    const response = await fetch(`${API_BASE}/api/analyze-xray`, { method: "POST", body: formData });
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server error (${response.status})`);
    }
    const data = await response.json();

    document.getElementById("heatmapImagePreview").src = `data:image/png;base64,${data.heatmap_image_base64}`;
    document.getElementById("xraySimplifiedText").textContent = data.simplified_text;
    renderFeaturesList(data.top_features || []);

    xrayResults.classList.remove("hidden");
    setBadge(xrayStatusBadge, "Analysis complete", "badge-success");
  } catch (err) {
    console.error(err);
    showToast(`X-ray analysis failed: ${err.message}`, true);
    setBadge(xrayStatusBadge, "Analysis failed", "badge-error");
    xrayEmptyState.classList.remove("hidden");
  } finally {
    xrayLoading.classList.add("hidden");
    analyzeXrayBtn.disabled = false;
  }
}

function initXrayActions() {
  analyzeXrayBtn.addEventListener("click", analyzeXray);

  document.getElementById("copyXrayBtn").addEventListener("click", () => {
    const text = document.getElementById("xraySimplifiedText").textContent;
    if (!text) return;
    navigator.clipboard.writeText(text)
      .then(() => showToast("Copied explanation to clipboard."))
      .catch(() => showToast("Could not copy to clipboard.", true));
  });

  document.getElementById("printXrayBtn").addEventListener("click", () => {
    const text = document.getElementById("xraySimplifiedText").textContent;
    if (!text) return;
    printSimpleReport("MedExplain — Plain-English Report", text);
  });
}

function printSimpleReport(title, bodyText) {
  const printWindow = window.open("", "_blank", "width=650,height=800");
  printWindow.document.write(`
    <html>
      <head><title>${title}</title></head>
      <body style="font-family: sans-serif; padding: 2rem; line-height: 1.6;">
        <h2>${title}</h2>
        <p>${bodyText.replace(/\n/g, "<br/>")}</p>
        <hr/>
        <p style="font-size: 0.8rem; color: #666;">
          Educational demo output — not a diagnostic report. Consult a licensed
          healthcare professional for medical decisions.
        </p>
      </body>
    </html>
  `);
  printWindow.document.close();
  printWindow.focus();
  printWindow.print();
}

/* ========================================================================
   TAB 2: BLOOD REPORT CHECKER
   ======================================================================== */

const hemoglobinInput = document.getElementById("hemoglobinInput");
const wbcInput = document.getElementById("wbcInput");
const plateletsInput = document.getElementById("plateletsInput");
const glucoseInput = document.getElementById("glucoseInput");
const analyzeBloodBtn = document.getElementById("analyzeBloodBtn");
const bloodStatusBadge = document.getElementById("bloodStatusBadge");
const bloodLoading = document.getElementById("bloodLoading");
const bloodEmptyState = document.getElementById("bloodEmptyState");
const bloodResults = document.getElementById("bloodResults");
const bloodCardsGrid = document.getElementById("bloodCardsGrid");
const bloodExtractedNotice = document.getElementById("bloodExtractedNotice");

const SAMPLE_BLOOD_CASES = {
  normal: { hemoglobin: 14.2, wbc: 6.8, platelets: 260, glucose: 88 },
  anemia: { hemoglobin: 9.4, wbc: 5.9, platelets: 210, glucose: 91 },
  prediabetic: { hemoglobin: 13.8, wbc: 7.4, platelets: 300, glucose: 112 },
};

function initSampleBloodButtons() {
  document.querySelectorAll(".sample-btn-blood").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = SAMPLE_BLOOD_CASES[btn.dataset.sample];
      if (!preset) return;
      hemoglobinInput.value = preset.hemoglobin;
      wbcInput.value = preset.wbc;
      plateletsInput.value = preset.platelets;
      glucoseInput.value = preset.glucose;
      showToast(`Loaded sample: ${btn.textContent.trim()}`);
    });
  });
}

function statusToClass(status) {
  const map = {
    Normal: "status-normal", Borderline: "status-borderline", Critical: "status-critical",
    "Low Risk": "status-normal", "Mild Risk": "status-borderline",
    "Moderate Risk": "status-borderline", "High Risk": "status-critical",
  };
  return map[status] || "badge-idle";
}

function renderBloodResults(results) {
  bloodCardsGrid.innerHTML = "";
  results.forEach((r) => {
    const isOverall = r.key === "overall_risk_index";
    const card = document.createElement("div");
    card.className = "biomarker-card" + (isOverall ? " overall" : "");
    card.innerHTML = `
      <div class="biomarker-card-header">
        <h4>${r.label}</h4>
        <span class="badge ${statusToClass(r.status)}">${r.status}</span>
      </div>
      <div class="biomarker-value">${r.value}<span>${r.unit}</span></div>
      <div class="biomarker-range">Reference range: ${r.normal_range}</div>
      <p class="biomarker-insight">${r.insight}</p>
    `;
    bloodCardsGrid.appendChild(card);
  });
}

async function analyzeBlood() {
  const payload = {};
  if (hemoglobinInput.value !== "") payload.hemoglobin = parseFloat(hemoglobinInput.value);
  if (wbcInput.value !== "") payload.wbc = parseFloat(wbcInput.value);
  if (plateletsInput.value !== "") payload.platelets = parseFloat(plateletsInput.value);
  if (glucoseInput.value !== "") payload.glucose = parseFloat(glucoseInput.value);

  if (Object.keys(payload).length === 0) {
    showToast("Please enter at least one lab value.", true);
    return;
  }

  bloodEmptyState.classList.add("hidden");
  bloodResults.classList.add("hidden");
  bloodExtractedNotice.classList.add("hidden");
  bloodLoading.classList.remove("hidden");
  analyzeBloodBtn.disabled = true;
  setBadge(bloodStatusBadge, "Analyzing…", "badge-warning");

  try {
    const response = await fetch(`${API_BASE}/api/analyze-blood`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server error (${response.status})`);
    }
    const data = await response.json();
    renderBloodResults(data.results);
    bloodResults.classList.remove("hidden");
    setBadge(bloodStatusBadge, "Analysis complete", "badge-success");
  } catch (err) {
    console.error(err);
    showToast(`Blood analysis failed: ${err.message}`, true);
    setBadge(bloodStatusBadge, "Analysis failed", "badge-error");
    bloodEmptyState.classList.remove("hidden");
  } finally {
    bloodLoading.classList.add("hidden");
    analyzeBloodBtn.disabled = false;
  }
}

/* --- Blood report document upload --- */
let currentBloodDocFile = null;
const bloodDropzone = document.getElementById("bloodDropzone");
const bloodFileInput = document.getElementById("bloodFileInput");
const bloodFileName = document.getElementById("bloodFileName");
const analyzeBloodDocBtn = document.getElementById("analyzeBloodDocBtn");

function handleSelectedBloodDoc(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg", "image/jpg", "image/webp", "application/pdf"].includes(file.type)) {
    showToast("Please choose a PNG, JPG, or PDF file.", true);
    return;
  }
  currentBloodDocFile = file;
  bloodFileName.textContent = `Selected: ${file.name}`;
  analyzeBloodDocBtn.disabled = false;
}

function initBloodDropzone() {
  bloodDropzone.addEventListener("click", () => bloodFileInput.click());
  bloodFileInput.addEventListener("change", (e) => handleSelectedBloodDoc(e.target.files[0]));
  ["dragenter", "dragover"].forEach((evt) => {
    bloodDropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); bloodDropzone.classList.add("dragover"); });
  });
  ["dragleave", "drop"].forEach((evt) => {
    bloodDropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); bloodDropzone.classList.remove("dragover"); });
  });
  bloodDropzone.addEventListener("drop", (e) => handleSelectedBloodDoc(e.dataTransfer.files && e.dataTransfer.files[0]));
}

async function analyzeBloodDocument() {
  if (!currentBloodDocFile) {
    showToast("Please upload a report file first.", true);
    return;
  }

  bloodEmptyState.classList.add("hidden");
  bloodResults.classList.add("hidden");
  bloodLoading.classList.remove("hidden");
  analyzeBloodDocBtn.disabled = true;
  setBadge(bloodStatusBadge, "Extracting…", "badge-warning");

  try {
    const formData = new FormData();
    formData.append("file", currentBloodDocFile);

    const response = await fetch(`${API_BASE}/api/analyze-blood-document`, { method: "POST", body: formData });
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server error (${response.status})`);
    }
    const data = await response.json();

    const extractedKeys = Object.keys(data.extracted_values || {});
    bloodExtractedNotice.textContent =
      `Extracted ${extractedKeys.length} value(s) from your document: ${extractedKeys.join(", ") || "none"}. ` +
      `Please double-check these against your original report before relying on them.`;
    bloodExtractedNotice.classList.remove("hidden");

    // Prefill the manual form too, so the user can review/correct.
    if (data.extracted_values.hemoglobin !== undefined) hemoglobinInput.value = data.extracted_values.hemoglobin;
    if (data.extracted_values.wbc !== undefined) wbcInput.value = data.extracted_values.wbc;
    if (data.extracted_values.platelets !== undefined) plateletsInput.value = data.extracted_values.platelets;
    if (data.extracted_values.glucose !== undefined) glucoseInput.value = data.extracted_values.glucose;

    renderBloodResults(data.results);
    bloodResults.classList.remove("hidden");
    setBadge(bloodStatusBadge, "Analysis complete", "badge-success");
  } catch (err) {
    console.error(err);
    showToast(`Document analysis failed: ${err.message}`, true);
    setBadge(bloodStatusBadge, "Analysis failed", "badge-error");
    bloodEmptyState.classList.remove("hidden");
  } finally {
    bloodLoading.classList.add("hidden");
    analyzeBloodDocBtn.disabled = false;
  }
}

function initBloodActions() {
  analyzeBloodBtn.addEventListener("click", analyzeBlood);
  analyzeBloodDocBtn.addEventListener("click", analyzeBloodDocument);
}

/* ========================================================================
   TAB 3: PRESCRIPTION ANALYSIS
   ======================================================================== */

let currentPrescriptionFile = null;
const prescriptionDropzone = document.getElementById("prescriptionDropzone");
const prescriptionFileInput = document.getElementById("prescriptionFileInput");
const prescriptionFileName = document.getElementById("prescriptionFileName");
const analyzePrescriptionBtn = document.getElementById("analyzePrescriptionBtn");
const prescriptionStatusBadge = document.getElementById("prescriptionStatusBadge");
const prescriptionLoading = document.getElementById("prescriptionLoading");
const prescriptionEmptyState = document.getElementById("prescriptionEmptyState");
const prescriptionResults = document.getElementById("prescriptionResults");

function handleSelectedPrescriptionFile(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg", "image/jpg", "image/webp", "application/pdf"].includes(file.type)) {
    showToast("Please choose a PNG, JPG, or PDF file.", true);
    return;
  }
  currentPrescriptionFile = file;
  prescriptionFileName.textContent = `Selected: ${file.name}`;
  analyzePrescriptionBtn.disabled = false;
  setBadge(prescriptionStatusBadge, "File ready", "badge-warning");
}

function initPrescriptionDropzone() {
  prescriptionDropzone.addEventListener("click", () => prescriptionFileInput.click());
  prescriptionFileInput.addEventListener("change", (e) => handleSelectedPrescriptionFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((evt) => {
    prescriptionDropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); prescriptionDropzone.classList.add("dragover"); });
  });
  ["dragleave", "drop"].forEach((evt) => {
    prescriptionDropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); prescriptionDropzone.classList.remove("dragover"); });
  });
  prescriptionDropzone.addEventListener("drop", (e) => handleSelectedPrescriptionFile(e.dataTransfer.files && e.dataTransfer.files[0]));
}

function initSamplePrescriptionButton() {
  document.querySelectorAll(".sample-btn-prescription").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const filename = btn.dataset.sample;
      try {
        const file = await fetchAsFile(`/static/samples/${filename}`, filename, "image/png");
        handleSelectedPrescriptionFile(file);
        showToast("Loaded sample prescription.");
      } catch (err) {
        showToast("Failed to load sample prescription.", true);
        console.error(err);
      }
    });
  });
}

function renderMedications(meds) {
  const grid = document.getElementById("medicationGrid");
  grid.innerHTML = "";
  if (!meds || meds.length === 0) {
    grid.innerHTML = `<p class="no-medications-note">No medications from our local reference list were recognized in this document. See the raw OCR text below for what was read.</p>`;
    return;
  }
  meds.forEach((m) => {
    const card = document.createElement("div");
    card.className = "medication-card";
    card.innerHTML = `
      <span class="medication-class">${m.drug_class}</span>
      <h5>${m.name}</h5>
      <p class="medication-use">${m.common_use}</p>
    `;
    grid.appendChild(card);
  });
}

async function analyzePrescription() {
  if (!currentPrescriptionFile) {
    showToast("Please upload a prescription file first.", true);
    return;
  }

  prescriptionEmptyState.classList.add("hidden");
  prescriptionResults.classList.add("hidden");
  prescriptionLoading.classList.remove("hidden");
  analyzePrescriptionBtn.disabled = true;
  setBadge(prescriptionStatusBadge, "Analyzing…", "badge-warning");

  try {
    const formData = new FormData();
    formData.append("file", currentPrescriptionFile);

    const response = await fetch(`${API_BASE}/api/analyze-prescription`, { method: "POST", body: formData });
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server error (${response.status})`);
    }
    const data = await response.json();

    renderMedications(data.identified_medications);
    document.getElementById("prescriptionSummaryText").textContent = data.plain_summary;
    document.getElementById("prescriptionRawText").textContent = data.raw_text || "(no text detected)";

    prescriptionResults.classList.remove("hidden");
    setBadge(prescriptionStatusBadge, "Analysis complete", "badge-success");
  } catch (err) {
    console.error(err);
    showToast(`Prescription analysis failed: ${err.message}`, true);
    setBadge(prescriptionStatusBadge, "Analysis failed", "badge-error");
    prescriptionEmptyState.classList.remove("hidden");
  } finally {
    prescriptionLoading.classList.add("hidden");
    analyzePrescriptionBtn.disabled = false;
  }
}

function initPrescriptionActions() {
  analyzePrescriptionBtn.addEventListener("click", analyzePrescription);

  document.getElementById("copyPrescriptionBtn").addEventListener("click", () => {
    const text = document.getElementById("prescriptionSummaryText").textContent;
    if (!text) return;
    navigator.clipboard.writeText(text)
      .then(() => showToast("Copied summary to clipboard."))
      .catch(() => showToast("Could not copy to clipboard.", true));
  });

  document.getElementById("printPrescriptionBtn").addEventListener("click", () => {
    const text = document.getElementById("prescriptionSummaryText").textContent;
    if (!text) return;
    printSimpleReport("MedExplain — Prescription Summary", text);
  });
}

/* ========================================================================
   TAB 4: SYMPTOM REASONER
   ======================================================================== */

const symptomInput = document.getElementById("symptomInput");
const analyzeSymptomBtn = document.getElementById("analyzeSymptomBtn");
const symptomStatusBadge = document.getElementById("symptomStatusBadge");
const symptomLoading = document.getElementById("symptomLoading");
const symptomEmptyState = document.getElementById("symptomEmptyState");
const symptomResults = document.getElementById("symptomResults");

function initSymptomChips() {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      symptomInput.value = chip.dataset.symptom;
      checkSymptom();
    });
  });
}

function renderList(elementId, items) {
  const list = document.getElementById(elementId);
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

async function checkSymptom() {
  const symptom = symptomInput.value.trim();
  if (!symptom) {
    showToast("Please enter a symptom first.", true);
    return;
  }

  symptomEmptyState.classList.add("hidden");
  symptomResults.classList.add("hidden");
  symptomLoading.classList.remove("hidden");
  analyzeSymptomBtn.disabled = true;
  setBadge(symptomStatusBadge, "Looking up…", "badge-warning");

  try {
    const response = await fetch(`${API_BASE}/api/symptom-check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptom }),
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server error (${response.status})`);
    }
    const data = await response.json();

    const titleEl = document.getElementById("symptomTitle");
    const messageEl = document.getElementById("symptomMessage");

    if (data.matched) {
      titleEl.textContent = data.matched_symptom;
      messageEl.classList.add("hidden");
    } else {
      titleEl.textContent = `"${data.query}"`;
      messageEl.textContent = data.message;
      messageEl.classList.remove("hidden");
    }

    renderList("symptomCausesList", data.common_causes);
    renderList("symptomSelfCareList", data.self_care);
    renderList("symptomRedFlagsList", data.red_flags);
    document.getElementById("symptomDisclaimer").textContent = data.disclaimer;

    symptomResults.classList.remove("hidden");
    setBadge(symptomStatusBadge, data.matched ? "Match found" : "General guidance", data.matched ? "badge-success" : "badge-warning");
  } catch (err) {
    console.error(err);
    showToast(`Symptom check failed: ${err.message}`, true);
    setBadge(symptomStatusBadge, "Lookup failed", "badge-error");
    symptomEmptyState.classList.remove("hidden");
  } finally {
    symptomLoading.classList.add("hidden");
    analyzeSymptomBtn.disabled = false;
  }
}

function initSymptomActions() {
  analyzeSymptomBtn.addEventListener("click", checkSymptom);
  symptomInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") checkSymptom();
  });
}

/* ---------------------------------------------------------------------- */
/* Init                                                                    */
/* ---------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  initNav();
  initSubTabs();

  initDropzone();
  initSampleXrayButtons();
  initXrayActions();

  initSampleBloodButtons();
  initBloodDropzone();
  initBloodActions();

  initPrescriptionDropzone();
  initSamplePrescriptionButton();
  initPrescriptionActions();

  initSymptomChips();
  initSymptomActions();
});
