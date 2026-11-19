// DermaScan review screen (DERM-16). Talks to the API in contract C3. No framework.
"use strict";

const MEANING = {
  urgent: "Suspected-cancer referral (seen within 2 weeks)",
  routine: "Non-urgent dermatology referral",
  monitor: "GP follow-up",
  cannot_assess: "Cannot assess this image. Refer for specialist review.",
};
const CLASS_NAMES = {
  akiec: "Actinic keratosis", bcc: "Basal cell carcinoma", bkl: "Benign keratosis",
  df: "Dermatofibroma", mel: "Melanoma", nv: "Melanocytic nevus", vasc: "Vascular lesion",
};

const $ = (id) => document.getElementById(id);
let reviewer = "";
let current = null; // last /api/assess response

// --- reviewer ID (kept for this browser tab only) -----------------------------------
function loadReviewer() {
  try { return sessionStorage.getItem("dermascan.reviewer") || ""; } catch (e) { return ""; }
}
function saveReviewer(id) {
  try { sessionStorage.setItem("dermascan.reviewer", id); } catch (e) { /* storage blocked: keep in memory */ }
}
function showReviewer() {
  const known = reviewer !== "";
  $("step-reviewer").hidden = known;
  $("reviewer-bar").hidden = !known;
  $("step-upload").hidden = !known;
  $("reviewer-name").textContent = reviewer;
}
$("reviewer-form").addEventListener("submit", (e) => {
  e.preventDefault();
  reviewer = $("reviewer-input").value.trim();
  if (!reviewer) return;
  saveReviewer(reviewer);
  showReviewer();
});
$("change-reviewer").addEventListener("click", () => {
  reviewer = "";
  $("reviewer-input").value = "";
  resetCase();
  showReviewer();
});

// --- helpers -------------------------------------------------------------------------
function showError(el, message) { el.textContent = message; el.hidden = !message; }

async function callApi(url, options) {
  let res;
  try { res = await fetch(url, options); } catch (e) { throw new Error("Cannot reach the DermaScan server."); }
  let body = null;
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) throw new Error((body && body.message) || `Request failed (${res.status})`);
  return body;
}

// --- upload and assess ---------------------------------------------------------------
$("image").addEventListener("change", () => {
  const file = $("image").files[0];
  const img = $("preview");
  if (!file) { img.hidden = true; return; }
  img.src = URL.createObjectURL(file);
  img.hidden = false;
});

$("assess-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError($("assess-error"), "");
  const file = $("image").files[0];
  if (!file) return showError($("assess-error"), "Choose an image first.");
  if (file.size > 10 * 1024 * 1024) return showError($("assess-error"), "Image is over 10 MB.");
  const form = new FormData();
  form.append("image", file);
  if ($("age").value !== "") form.append("age", $("age").value);
  if ($("site").value !== "") form.append("site", $("site").value);
  form.append("clinician", reviewer);
  const button = $("assess-button");
  button.disabled = true; button.textContent = "Assessing...";
  try {
    current = await callApi("/api/assess", { method: "POST", body: form });
    showResult(current);
  } catch (err) {
    showError($("assess-error"), err.message);
  } finally {
    button.disabled = false; button.textContent = "Assess";
  }
});

function showResult(r) {
  const { c1, c2 } = r;
  $("step-result").hidden = false;
  $("mock-note").hidden = !String(c1.model_version || "").startsWith("mock-");
  const badge = $("urgency-badge");
  badge.className = `badge ${c2.urgency}`;
  badge.textContent = c2.urgency.replace("_", " ");
  $("urgency-meaning").textContent = MEANING[c2.urgency] || "";
  $("score-line").textContent = c2.score === null ? "" : `Urgency score ${c2.score} out of 100`;

  const reasons = $("reasons");
  reasons.replaceChildren();
  c2.fired_rules.forEach((rule) => {
    if (rule.kind === "floor" && !rule.applied) return; // weak safety rules that did not change the result
    const li = document.createElement("li");
    li.textContent = rule.explanation + " ";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = `(${rule.id}, strength ${rule.strength})`;
    li.appendChild(tag);
    reasons.appendChild(li);
  });

  const unc = c1.entropy;
  const level = unc >= 0.55 ? "high" : unc >= 0.3 ? "medium" : "low";
  $("uncertainty-line").textContent = `Uncertainty: ${level} (normalised entropy ${unc}).`;
  const table = $("probs-table");
  table.replaceChildren();
  Object.entries(c1.probs).sort((a, b) => b[1] - a[1]).forEach(([cls, p]) => {
    const tr = table.insertRow();
    tr.insertCell().textContent = CLASS_NAMES[cls] || cls;
    const barCell = tr.insertCell();
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = `${Math.round(p * 100)}%`;
    barCell.appendChild(bar);
    tr.insertCell().textContent = `${(p * 100).toFixed(1)}%`;
  });
  const heat = $("heatmap");
  if (c1.heatmap_png_b64) {
    heat.src = `data:image/png;base64,${c1.heatmap_png_b64}`;
    heat.hidden = false;
    $("heatmap-note").textContent = "Heatmap: areas the model relied on most.";
  } else {
    heat.hidden = true;
    $("heatmap-note").textContent = "Heatmap not available yet (arrives with the real model).";
  }
  $("versions").textContent = `Model ${c1.model_version} · Rules ${c2.rules_version} · Case ${r.case_id}`;

  // decision: pre-select the recommendation; cannot_assess has none
  document.querySelectorAll("input[name=final]").forEach((el) => {
    el.checked = el.value === c2.urgency;
    el.disabled = false;
  });
  $("reason").value = "";
  $("reason").disabled = false;
  $("decision-form").querySelector("button").disabled = false;
  showError($("decision-error"), "");
  $("saved").hidden = true;
  $("new-case").hidden = true;
  updateReasonHint();
  $("step-result").scrollIntoView({ behavior: "smooth" });
}

// --- decision ------------------------------------------------------------------------
function chosen() {
  const el = document.querySelector("input[name=final]:checked");
  return el ? el.value : "";
}
function reasonRequired() {
  return current && chosen() !== current.c2.urgency; // always true for cannot_assess
}
function updateReasonHint() {
  const required = reasonRequired();
  $("reason-label").textContent = required ? "Reason (required)" : "Reason (optional)";
  $("reason-hint").textContent = required
    ? "Your decision differs from the recommendation, so a reason is required for the audit log."
    : "";
}
document.querySelectorAll("input[name=final]").forEach((el) => el.addEventListener("change", updateReasonHint));

$("decision-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError($("decision-error"), "");
  const final = chosen();
  const reason = $("reason").value.trim();
  if (!final) return showError($("decision-error"), "Choose a decision.");
  if (reasonRequired() && !reason) return showError($("decision-error"), "A reason is required.");
  try {
    const record = await callApi(`/api/cases/${encodeURIComponent(current.case_id)}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ final_decision: final, override_reason: reason || null, clinician: reviewer }),
    });
    const overridden = record.final_decision !== record.recommended_urgency;
    const saved = $("saved");
    saved.textContent = `Saved to the case log: ${record.final_decision}` +
      (overridden ? ` (override of ${record.recommended_urgency.replace("_", " ")})` : " (confirmed)") +
      ` by ${record.clinician} at ${record.decided_at} UTC.`;
    saved.hidden = false;
    document.querySelectorAll("input[name=final]").forEach((el) => { el.disabled = true; });
    $("reason").disabled = true;
    $("decision-form").querySelector("button").disabled = true;
    $("new-case").hidden = false;
  } catch (err) {
    showError($("decision-error"), err.message);
  }
});

function resetCase() {
  current = null;
  $("assess-form").reset();
  $("preview").hidden = true;
  $("step-result").hidden = true;
  showError($("assess-error"), "");
}
$("new-case").addEventListener("click", () => { resetCase(); window.scrollTo({ top: 0 }); });

reviewer = loadReviewer();
showReviewer();
