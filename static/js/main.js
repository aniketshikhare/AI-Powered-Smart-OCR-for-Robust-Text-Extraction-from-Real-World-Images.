"use strict";

const form = document.getElementById("ocr-form");
const input = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const preview = document.getElementById("preview");
const output = document.getElementById("output");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const metrics = document.getElementById("metrics");
const evaluation = document.getElementById("evaluation");
const details = document.getElementById("details");
const submitBtn = document.getElementById("submit-btn");

let lastResult = null;

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropzoneText.hidden = true;
  };
  reader.readAsDataURL(file);
}

dropzone.addEventListener("click", () => input.click());
input.addEventListener("change", () => input.files[0] && showPreview(input.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    input.files = e.dataTransfer.files;
    showPreview(file);
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  if (!input.files[0]) {
    errorEl.textContent = "Please select an image first.";
    errorEl.hidden = false;
    return;
  }

  const data = new FormData(form);
  data.set("image", input.files[0]);
  form.querySelectorAll("input[type=checkbox]").forEach((cb) => data.set(cb.name, cb.checked ? "1" : "0"));

  submitBtn.disabled = true;
  statusEl.textContent = "Running pipeline: preprocessing → detection → recognition → post-processing…";
  evaluation.hidden = true;

  try {
    const res = await fetch("/api/ocr", { method: "POST", body: data });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || "Extraction failed.");
    lastResult = payload;
    renderResult(payload);
  } catch (err) {
    statusEl.textContent = "Extraction failed.";
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
  }
});

function renderResult(r) {
  output.value = r.text || "(no text detected)";
  statusEl.textContent = `OCR ID ${r.ocr_id} · engine ${r.engine} · quality ${r.quality_label}`;
  document.getElementById("m-conf").textContent = `${r.confidence.toFixed(1)}%`;
  document.getElementById("m-words").textContent = r.word_count;
  document.getElementById("m-regions").textContent = r.region_count;
  document.getElementById("m-time").textContent = `${r.elapsed_ms} ms`;
  metrics.hidden = false;

  if (r.evaluation) {
    const e = r.evaluation;
    document.getElementById("m-char-acc").textContent = `${e.character_accuracy.toFixed(1)}%`;
    document.getElementById("m-word-acc").textContent = `${e.word_accuracy.toFixed(1)}%`;
    document.getElementById("m-cer").textContent = `${e.cer.toFixed(1)}%`;
    document.getElementById("m-wer").textContent = `${e.wer.toFixed(1)}%`;
    evaluation.hidden = false;
  }

  details.hidden = false;
  document.getElementById("details-body").textContent = JSON.stringify(
    {
      preprocess_steps: r.preprocess_steps,
      postprocess_steps: r.postprocess_steps,
      skew_angle: r.skew_angle,
      removed_tokens: r.removed_tokens,
      corrections: r.corrections,
      evaluation: r.evaluation || null,
    },
    null,
    2
  );
  refreshHistory();
}

document.getElementById("copy-btn").addEventListener("click", async () => {
  await navigator.clipboard.writeText(output.value);
  statusEl.textContent = "Copied to clipboard.";
});

document.getElementById("download-btn").addEventListener("click", () => {
  if (lastResult) window.location = `/download/${lastResult.ocr_id}.txt`;
});
document.getElementById("json-btn").addEventListener("click", () => {
  if (lastResult) window.location = `/download/${lastResult.ocr_id}.json`;
});

async function refreshHistory() {
  const res = await fetch("/api/records");
  const rows = await res.json();
  const body = document.getElementById("history-body");
  body.innerHTML = rows
    .slice(0, 10)
    .map(
      (r) =>
        `<tr><td>${r.ocr_id}</td><td>${r.image_name}</td><td>${r.confidence_score.toFixed(1)}%</td>` +
        `<td>${r.created_at.replace("T", " ")}</td><td><a href="/download/${r.ocr_id}.txt">txt</a></td></tr>`
    )
    .join("");
}
