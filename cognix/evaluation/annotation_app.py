"""Local Human-Reviewable 2D Object Annotation Application (Milestones 8, 9, 10, 12, AI-Assisted).

Lightweight Flask web application serving an interactive HTML5 Canvas tool
for annotating 2D bounding boxes and temporal track IDs on recorded CARLA frames.

Guiding Principles:
1. Dual Workflow Modes:
   - AI-Assisted Annotation: Faster R-CNN MobileNetV3 proposals are rendered in a distinct
     dashed layer with confidence scores to accelerate annotation. Reviewers accept, correct,
     or reject proposals, and manually add missing obstacles (e.g. road debris/crates).
   - Independent Ground-Truth Review: Predictions and proposals are completely hidden so the
     reviewer produces unbiased, independent ground truth suitable for formal scientific scoring.
2. Strict Provenance Tracking:
   - Every object records its exact provenance: 'ai_accepted', 'human_corrected', 'human_added',
     or 'independent_human'. Datasets produced with AI assistance are explicitly tagged so they
     cannot be misrepresented as independent benchmark ground truth.
3. Defensible Schema: Strictly enforces xyxy pixel coordinates, unique track_ids
   per frame, explicit class selection (no default to person), and formal confirmation statuses.
4. Zero External Downloads: Pure HTML5 Canvas and vanilla JavaScript running on
   local Flask; requires zero external npm or CDN dependencies.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_file, render_template_string

from cognix.evaluation.ground_truth_schema import (
    AnnotationDataset,
    FrameAnnotation,
    AnnotatedObject,
    BoundingBox,
    ObjectCategory,
    OcclusionStatus,
    VisibilityStatus,
    ConfirmationStatus,
    ValidationError,
)
from cognix.evaluation.tracking_evaluator import (
    map_detector_class_to_taxonomy,
    COCO_TO_EVALUATION_TAXONOMY,
    UNSUPPORTED_CLASS_CATEGORY,
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cognix M8 Annotation & Review Studio</title>
<style>
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #1a1e24;
    color: #e0e6ed;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  header {
    background-color: #242b35;
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #384252;
  }
  header h1 {
    font-size: 18px;
    margin: 0;
    color: #4da6ff;
  }
  .status-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 13px;
  }
  .badge-pending { background: #d97706; color: #fff; }
  .badge-confirmed { background: #16a34a; color: #fff; }
  .badge-ai { background: #0891b2; color: #fff; }
  .badge-independent { background: #7c3aed; color: #fff; }
  .main-container {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  .canvas-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background-color: #12151a;
    position: relative;
    overflow: auto;
    padding: 10px;
  }
  #annotationCanvas {
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    cursor: crosshair;
    background-color: #000;
    max-width: 100%;
    max-height: 85vh;
  }
  .sidebar {
    width: 400px;
    background-color: #242b35;
    border-left: 1px solid #384252;
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .control-group {
    background: #1c222a;
    border: 1px solid #313b49;
    border-radius: 6px;
    padding: 12px;
  }
  .control-group h3 {
    margin: 0 0 10px 0;
    font-size: 14px;
    color: #8bb4e7;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  label {
    display: block;
    font-size: 12px;
    margin-bottom: 4px;
    color: #9cb0c9;
  }
  select, input, button {
    width: 100%;
    padding: 8px 10px;
    border-radius: 4px;
    border: 1px solid #3d4a5d;
    background: #2a3340;
    color: #fff;
    font-size: 13px;
    box-sizing: border-box;
    margin-bottom: 8px;
  }
  select:focus, input:focus {
    border-color: #4da6ff;
    outline: none;
  }
  button {
    cursor: pointer;
    background: #2563eb;
    font-weight: 600;
    border: none;
    transition: background 0.15s ease;
  }
  button:hover { background: #1d4ed8; }
  button.danger { background: #dc2626; }
  button.danger:hover { background: #b91c1c; }
  button.success { background: #16a34a; }
  button.success:hover { background: #15803d; }
  button.secondary { background: #475569; }
  button.secondary:hover { background: #334155; }
  .btn-row {
    display: flex;
    gap: 8px;
  }
  .btn-row button {
    flex: 1;
  }
  .object-item {
    padding: 8px;
    background: #2a3340;
    border-radius: 4px;
    margin-bottom: 6px;
    cursor: pointer;
    border-left: 4px solid #4da6ff;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .object-item.selected {
    background: #364457;
    border-color: #38bdf8;
  }
  .proposal-item {
    padding: 6px 8px;
    background: #1e293b;
    border-radius: 4px;
    margin-bottom: 6px;
    border-left: 4px dashed #06b6d4;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .footer-bar {
    background: #1c222a;
    border-top: 1px solid #384252;
    padding: 6px 20px;
    font-size: 12px;
    color: #7b8ea6;
    display: flex;
    justify-content: space-between;
  }
  .provenance-tag {
    font-size: 10px;
    padding: 2px 5px;
    border-radius: 3px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .prov-ai { background: #0891b2; color: #fff; }
  .prov-corrected { background: #d97706; color: #fff; }
  .prov-human { background: #16a34a; color: #fff; }
  .prov-independent { background: #7c3aed; color: #fff; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Cognix M8 Annotation & Review Studio</h1>
    <span id="headerSubtext" style="font-size: 12px; color: #8bb4e7;">AI-Assisted Annotation & Proposal Review</span>
  </div>
  <div style="display:flex; align-items:center; gap: 12px;">
    <div id="modeBadge" class="status-badge badge-ai">AI-ASSISTED</div>
    <div id="queueProgress" style="font-size: 13px;">Progress: Loading...</div>
    <div id="frameStatusBadge" class="status-badge badge-pending">PENDING REVIEW</div>
  </div>
</header>

<div class="main-container">
  <div class="canvas-panel">
    <canvas id="annotationCanvas" width="1920" height="1080"></canvas>
  </div>

  <div class="sidebar">
    <div class="control-group">
      <h3>Workflow & Navigation</h3>
      <label>Workflow Mode:</label>
      <select id="modeSelect">
        <option value="ai_assisted">🤖 AI-Assisted Mode (Model Proposals Active)</option>
        <option value="independent">👤 Independent Ground-Truth Review (Proposals Hidden)</option>
      </select>

      <label>Scenario:</label>
      <select id="scenarioSelect"></select>

      <label>Frame (Tick):</label>
      <div class="btn-row">
        <button id="prevFrameBtn">&larr; Prev (A)</button>
        <button id="nextFrameBtn">Next (D) &rarr;</button>
      </div>
      <select id="tickSelect"></select>
    </div>

    <!-- AI Detection Proposals Panel (visible only in AI-Assisted Mode) -->
    <div class="control-group" id="proposalsPanel">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <h3 style="margin:0;">AI Proposals (<span id="propCount">0</span>)</h3>
        <span id="propSourceBadge" style="font-size:11px; padding:2px 6px; border-radius:3px; background:#0f172a; color:#38bdf8;">Cached</span>
      </div>

      <div style="margin: 4px 0 8px 0;">
        <input type="checkbox" id="toggleProposalsLayer" checked style="width: auto; margin-right: 6px;">
        <label for="toggleProposalsLayer" style="display:inline; font-weight:600; color:#38bdf8;">Show Proposals Layer (Dashed Cyan)</label>
      </div>

      <div id="aiLimitationNotice" style="font-size:11px; color:#f59e0b; background:#291e0a; border:1px solid #78350f; border-radius:4px; padding:6px; margin-bottom:8px; line-height:1.4;">
        &#9888; <b>Detector Scope:</b> Pretrained COCO Faster R-CNN detects people & vehicles. Custom roadway crates/props are not in COCO. Draw missed props manually.
      </div>

      <button id="acceptAllProposalsBtn" style="background:#0284c7; margin-bottom:8px;">Accept All Unmatched Proposals</button>
      <div id="proposalsList" style="max-height: 140px; overflow-y:auto;"></div>
    </div>

    <div class="control-group">
      <h3>Selected Object</h3>
      <div id="noObjectNotice" style="font-size:12px; color:#94a3b8;">Click & drag on image to draw a bounding box, or click an object to edit.</div>
      <div id="objectForm" style="display:none;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:12px; color:#94a3b8;">Object Origin / Provenance:</span>
          <span id="objProvenanceBadge" class="provenance-tag prov-human">Human</span>
        </div>

        <label>Track ID (persistent across sequence):</label>
        <input type="number" id="objTrackId" min="0" step="1">

        <label>Object Class (Required):</label>
        <select id="objCategory">
          <option value="" disabled selected>-- Select Object Class (Required) --</option>
          <option value="person">Person / Pedestrian</option>
          <option value="vehicle">Vehicle (Car/Truck)</option>
          <option value="bicycle_or_motorcycle">Bicycle / Motorcycle</option>
          <option value="road_debris_or_prop">Road Debris / Spawned Prop</option>
          <option value="ignore_ambiguous">Ignore / Ambiguous Region</option>
        </select>
        <div id="classRequiredAlert" style="display:none; color:#f43f5e; font-size:12px; margin-bottom:8px; font-weight:600;">&#9888; Please select an object class from the dropdown above.</div>

        <label>Occlusion:</label>
        <select id="objOcclusion">
          <option value="none">None (Fully visible)</option>
          <option value="partial">Partial (10-50% obscured)</option>
          <option value="heavy">Heavy (&gt; 50% obscured)</option>
        </select>

        <label>Visibility:</label>
        <select id="objVisibility">
          <option value="clear">Clear</option>
          <option value="degraded_weather">Degraded (Rain/Fog)</option>
          <option value="glare_or_shadow">Glare / Shadow</option>
        </select>

        <div style="margin: 8px 0;">
          <input type="checkbox" id="objAmbiguous" style="width: auto; margin-right: 6px;">
          <label for="objAmbiguous" style="display:inline;">Mark Ambiguous / Low Confidence</label>
        </div>

        <button id="deleteObjBtn" class="danger">Delete Object</button>
      </div>
    </div>

    <div class="control-group">
      <h3>Annotations in Frame (<span id="objCount">0</span>)</h3>
      <div id="objectList" style="max-height: 150px; overflow-y:auto;"></div>
    </div>

    <div class="control-group">
      <h3>Review & Confirmation</h3>
      <label>Reviewer Name / ID:</label>
      <input type="text" id="reviewerId" placeholder="e.g. Aditya (Safety Auditor)">

      <label>Review Notes (Optional):</label>
      <input type="text" id="reviewNotes" placeholder="Notes on clutter, props, lighting...">

      <button id="confirmFrameBtn" class="success">Confirm & Next Frame</button>
      <button id="saveDatasetBtn" style="margin-top:6px;">Save Annotations to Disk</button>
    </div>
  </div>
</div>

<div class="footer-bar">
  <div>Image: <span id="imageDimensions">1920 &times; 1080</span> | SHA-256: <code id="frameHash">Loading...</code></div>
  <div>Dataset SHA-256: <code id="datasetHash">Loading...</code></div>
</div>

<script>
let currentScenario = "";
let currentTick = 0;
let currentFrameData = null;
let dataset = null;
let currentImage = new Image();
let selectedObjIndex = -1;
let isDrawing = false;
let startX = 0, startY = 0;
let imageLoading = false;
let imageError = null;

// AI Proposals State
let currentProposals = [];
let proposalSource = "none";
let showProposalsLayer = true;
let currentMode = "ai_assisted";

const canvas = document.getElementById("annotationCanvas");
const ctx = canvas.getContext("2d");

async function init() {
  const resp = await fetch("/api/annotations");
  dataset = await resp.json();

  // Set mode from dataset or default
  currentMode = dataset.annotation_mode || "ai_assisted";
  document.getElementById("modeSelect").value = currentMode;
  updateModeUI();

  // Populate scenario select
  const scenarios = [...new Set(dataset.frames.map(f => f.scenario_id))];
  const scSelect = document.getElementById("scenarioSelect");
  scSelect.innerHTML = "";
  scenarios.forEach(sc => {
    const opt = document.createElement("option");
    opt.value = sc;
    opt.textContent = sc.split("/").slice(-2).join("/");
    scSelect.appendChild(opt);
  });

  currentScenario = scenarios[0];
  scSelect.value = currentScenario;
  scSelect.addEventListener("change", (e) => {
    currentScenario = e.target.value;
    updateTickList();
  });

  document.getElementById("tickSelect").addEventListener("change", (e) => {
    currentTick = parseInt(e.target.value);
    loadCurrentFrame();
  });

  document.getElementById("prevFrameBtn").addEventListener("click", () => navigateFrame(-1));
  document.getElementById("nextFrameBtn").addEventListener("click", () => navigateFrame(1));

  document.getElementById("modeSelect").addEventListener("change", (e) => {
    currentMode = e.target.value;
    dataset.annotation_mode = currentMode;
    updateModeUI();
    loadProposalsForCurrentFrame();
    redraw();
  });

  document.getElementById("toggleProposalsLayer").addEventListener("change", (e) => {
    showProposalsLayer = e.target.checked;
    redraw();
  });

  document.getElementById("acceptAllProposalsBtn").addEventListener("click", acceptAllUnmatchedProposals);

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "a" || e.key === "ArrowLeft") navigateFrame(-1);
    if (e.key === "d" || e.key === "ArrowRight") navigateFrame(1);
    if (e.key === "Delete" || e.key === "Backspace") deleteSelectedObject();
  });

  setupCanvasEvents();
  setupObjectFormEvents();
  updateTickList();
}

function updateModeUI() {
  const badge = document.getElementById("modeBadge");
  const panel = document.getElementById("proposalsPanel");
  const subtext = document.getElementById("headerSubtext");

  if (currentMode === "independent") {
    badge.textContent = "INDEPENDENT GT";
    badge.className = "status-badge badge-independent";
    panel.style.display = "none";
    subtext.textContent = "Independent Ground-Truth Review (Unbiased, No Proposals)";
    showProposalsLayer = false;
  } else {
    badge.textContent = "AI-ASSISTED";
    badge.className = "status-badge badge-ai";
    panel.style.display = "block";
    subtext.textContent = "AI-Assisted Annotation & Proposal Review";
    showProposalsLayer = document.getElementById("toggleProposalsLayer").checked;
  }
}

function updateTickList() {
  const scFrames = dataset.frames.filter(f => f.scenario_id === currentScenario);
  scFrames.sort((a,b) => a.tick - b.tick);

  const tickSelect = document.getElementById("tickSelect");
  tickSelect.innerHTML = "";
  scFrames.forEach(f => {
    const opt = document.createElement("option");
    opt.value = f.tick;
    const isConf = f.reviewer_id && (!f.objects || f.objects.length === 0 || f.objects.every(o => o.confirmation_status === "human_confirmed"));
    const statusText = isConf ? " [CONFIRMED]" : " [PENDING]";
    opt.textContent = `Tick ${f.tick}${statusText}`;
    tickSelect.appendChild(opt);
  });

  currentTick = scFrames[0].tick;
  tickSelect.value = currentTick;
  loadCurrentFrame();
}

function navigateFrame(dir) {
  const scFrames = dataset.frames.filter(f => f.scenario_id === currentScenario).sort((a,b) => a.tick - b.tick);
  const curIdx = scFrames.findIndex(f => f.tick === currentTick);
  const nextIdx = curIdx + dir;
  if (nextIdx >= 0 && nextIdx < scFrames.length) {
    currentTick = scFrames[nextIdx].tick;
    document.getElementById("tickSelect").value = currentTick;
    loadCurrentFrame();
  }
}

function loadCurrentFrame() {
  currentFrameData = dataset.frames.find(f => f.scenario_id === currentScenario && f.tick === currentTick);
  selectedObjIndex = -1;
  updateSidebar();

  const imgUrl = `/api/frame_image?scenario=${encodeURIComponent(currentScenario)}&tick=${currentTick}`;
  imageLoading = true;
  imageError = null;
  redraw();

  currentImage = new Image();
  currentImage.onload = () => {
    imageLoading = false;
    imageError = null;
    redraw();
  };
  currentImage.onerror = () => {
    imageLoading = false;
    imageError = `Failed to load frame image for scenario '${currentScenario}' at tick ${currentTick}`;
    redraw();
  };
  currentImage.src = imgUrl;

  document.getElementById("frameHash").textContent = currentFrameData ? (currentFrameData.frame_hash_sha256.substring(0, 16) + "...") : "N/A";
  updateProgressBadge();

  if (currentMode === "ai_assisted") {
    loadProposalsForCurrentFrame();
  } else {
    currentProposals = [];
    renderProposalsList();
  }
}

async function loadProposalsForCurrentFrame() {
  if (currentMode !== "ai_assisted") {
    currentProposals = [];
    renderProposalsList();
    return;
  }
  try {
    const url = `/api/proposals?scenario=${encodeURIComponent(currentScenario)}&tick=${currentTick}`;
    const resp = await fetch(url);
    if (resp.ok) {
      const data = await resp.json();
      currentProposals = data.proposals || [];
      proposalSource = data.source || "unknown";
      document.getElementById("propSourceBadge").textContent = proposalSource.replace("_", " ").toUpperCase();
      renderProposalsList();
      redraw();
    } else {
      currentProposals = [];
      renderProposalsList();
    }
  } catch (e) {
    currentProposals = [];
    renderProposalsList();
  }
}

function computeIoU(b1, b2) {
  const xA = Math.max(b1.x1, b2.x1);
  const yA = Math.max(b1.y1, b2.y1);
  const xB = Math.min(b1.x2, b2.x2);
  const yB = Math.min(b1.y2, b2.y2);
  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
  const area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
  const unionArea = area1 + area2 - interArea;
  return unionArea <= 0 ? 0 : (interArea / unionArea);
}

function findMatchingAnnotation(bbox) {
  if (!currentFrameData || !currentFrameData.objects) return null;
  for (let i = 0; i < currentFrameData.objects.length; i++) {
    const obj = currentFrameData.objects[i];
    if (computeIoU(bbox, obj.bbox) >= 0.50) {
      return { index: i, object: obj };
    }
  }
  return null;
}

function renderProposalsList() {
  const listEl = document.getElementById("proposalsList");
  const countEl = document.getElementById("propCount");
  listEl.innerHTML = "";
  countEl.textContent = currentProposals.length;

  if (currentProposals.length === 0) {
    listEl.innerHTML = "<div style='font-size:11px; color:#94a3b8; padding:4px;'>No proposals for this frame.</div>";
    return;
  }

  currentProposals.forEach((prop, pIdx) => {
    const match = findMatchingAnnotation(prop.bbox);
    const div = document.createElement("div");
    div.className = "proposal-item";

    let matchHtml = "";
    if (match) {
      matchHtml = `<span style="font-size:10px; color:#38bdf8; background:#0f172a; padding:2px 4px; border-radius:3px;">Matches ID ${match.object.track_id}</span>`;
    }

    const confPct = (prop.score * 100).toFixed(0);
    div.innerHTML = `
      <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:200px;">
        <b>${prop.category}</b> <span style="color:#aaa;">(${confPct}%)</span>
        <div>${matchHtml}</div>
      </div>
      <div style="display:flex; gap:4px;">
        ${match ? `<button class="secondary" style="padding:3px 6px; font-size:11px;" onclick="selectAnnotation(${match.index})">View</button>` : `<button class="success" style="padding:3px 6px; font-size:11px;" onclick="acceptProposal(${pIdx})">Accept</button>`}
        <button class="danger" style="padding:3px 6px; font-size:11px;" onclick="dismissProposal(${pIdx})">&times;</button>
      </div>
    `;
    listEl.appendChild(div);
  });
}

window.selectAnnotation = function(idx) {
  selectedObjIndex = idx;
  updateSidebar();
  redraw();
};

window.acceptProposal = function(pIdx) {
  if (pIdx < 0 || pIdx >= currentProposals.length) return;
  const prop = currentProposals[pIdx];

  // Ensure no duplicate track ID in this frame
  const nextTrackId = getUniqueTrackId(prop.track_id);
  const newObj = {
    object_id: `obj_${Date.now()}_${nextTrackId}`,
    track_id: nextTrackId,
    category: prop.category,
    bbox: {
      x1: Math.round(prop.bbox.x1),
      y1: Math.round(prop.bbox.y1),
      x2: Math.round(prop.bbox.x2),
      y2: Math.round(prop.bbox.y2)
    },
    occlusion: "none",
    visibility: "clear",
    is_ambiguous: false,
    confirmation_status: "pending_review",
    review_notes: "",
    confidence: prop.score,
    provenance: "ai_accepted"
  };

  if (!currentFrameData.objects) currentFrameData.objects = [];
  currentFrameData.objects.push(newObj);
  selectedObjIndex = currentFrameData.objects.length - 1;
  updateSidebar();
  renderProposalsList();
  redraw();
};

window.dismissProposal = function(pIdx) {
  if (pIdx >= 0 && pIdx < currentProposals.length) {
    currentProposals.splice(pIdx, 1);
    renderProposalsList();
    redraw();
  }
};

function acceptAllUnmatchedProposals() {
  let addedCount = 0;
  currentProposals.forEach(prop => {
    const match = findMatchingAnnotation(prop.bbox);
    if (!match) {
      const nextTrackId = getUniqueTrackId(prop.track_id);
      const newObj = {
        object_id: `obj_${Date.now()}_${nextTrackId}_${addedCount}`,
        track_id: nextTrackId,
        category: prop.category,
        bbox: {
          x1: Math.round(prop.bbox.x1),
          y1: Math.round(prop.bbox.y1),
          x2: Math.round(prop.bbox.x2),
          y2: Math.round(prop.bbox.y2)
        },
        occlusion: "none",
        visibility: "clear",
        is_ambiguous: false,
        confirmation_status: "pending_review",
        review_notes: "",
        confidence: prop.score,
        provenance: "ai_accepted"
      };
      if (!currentFrameData.objects) currentFrameData.objects = [];
      currentFrameData.objects.push(newObj);
      addedCount++;
    }
  });

  if (addedCount > 0) {
    selectedObjIndex = currentFrameData.objects.length - 1;
    updateSidebar();
    renderProposalsList();
    redraw();
  } else {
    alert("All proposals already match existing annotations or none available.");
  }
}

function getUniqueTrackId(preferredId = 1) {
  if (!currentFrameData || !currentFrameData.objects || currentFrameData.objects.length === 0) return preferredId;
  const existing = new Set(currentFrameData.objects.map(o => o.track_id));
  if (!existing.has(preferredId) && preferredId > 0) return preferredId;
  let next = 1;
  while (existing.has(next)) next++;
  return next;
}

function updateProgressBadge() {
  const total = dataset.frames.length;
  const confirmed = dataset.frames.filter(f => f.reviewer_id && (!f.objects || f.objects.length === 0 || f.objects.every(o => o.confirmation_status === "human_confirmed"))).length;
  document.getElementById("queueProgress").textContent = `Confirmed: ${confirmed} / ${total} frames`;

  const badge = document.getElementById("frameStatusBadge");
  const isCurConfirmed = currentFrameData && currentFrameData.reviewer_id && (!currentFrameData.objects || currentFrameData.objects.length === 0 || currentFrameData.objects.every(o => o.confirmation_status === "human_confirmed"));
  if (isCurConfirmed) {
    badge.textContent = "HUMAN CONFIRMED";
    badge.className = "status-badge badge-confirmed";
  } else {
    badge.textContent = "PENDING REVIEW";
    badge.className = "status-badge badge-pending";
  }
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (imageLoading) {
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#38bdf8";
    ctx.font = "bold 32px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`Loading Frame (Tick ${currentTick})...`, canvas.width / 2, canvas.height / 2 - 15);
    ctx.font = "18px sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(currentScenario, canvas.width / 2, canvas.height / 2 + 35);
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
    return;
  }

  if (imageError) {
    ctx.fillStyle = "#1a1e24";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ef4444";
    ctx.font = "bold 32px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Failed to Load Frame Image", canvas.width / 2, canvas.height / 2 - 20);
    ctx.fillStyle = "#f87171";
    ctx.font = "18px sans-serif";
    ctx.fillText(imageError, canvas.width / 2, canvas.height / 2 + 25);
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
    return;
  }

  if (currentImage.complete && currentImage.naturalWidth > 0) {
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "24px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No Image Available", canvas.width / 2, canvas.height / 2);
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }

  // Draw AI Proposals Layer (Dashed cyan) if active
  if (currentMode === "ai_assisted" && showProposalsLayer && currentProposals && currentProposals.length > 0) {
    currentProposals.forEach(prop => {
      const b = prop.bbox;
      const match = findMatchingAnnotation(b);

      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = match ? "#0e7490" : "#06b6d4";
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      ctx.setLineDash([]);

      const confPct = (prop.score * 100).toFixed(0);
      const tag = match ? `[AI: ${prop.category} ${confPct}% (Matched ID ${match.object.track_id})]` : `[AI: ${prop.category} ${confPct}%]`;

      ctx.fillStyle = match ? "#0e7490" : "#06b6d4";
      ctx.font = "12px sans-serif";
      const textWidth = ctx.measureText(tag).width;
      ctx.fillRect(b.x1, Math.max(0, b.y1 - 18), textWidth + 6, 18);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(tag, b.x1 + 3, Math.max(14, b.y1 - 4));
    });
  }

  // Draw Annotated Objects (Solid lines)
  if (currentFrameData && currentFrameData.objects) {
    currentFrameData.objects.forEach((obj, idx) => {
      const b = obj.bbox;
      const isSelected = (idx === selectedObjIndex);
      const isUnassigned = (!obj.category || obj.category.trim() === "");

      let strokeColor = "#f43f5e"; // Rose/red for unassigned
      if (!isUnassigned) {
        strokeColor = (obj.category === "person" ? "#22c55e" : (obj.category === "vehicle" ? "#3b82f6" : (obj.category === "bicycle_or_motorcycle" ? "#a855f7" : "#f59e0b")));
      }

      ctx.lineWidth = isSelected ? 4 : 2;
      ctx.strokeStyle = isSelected ? "#38bdf8" : strokeColor;
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

      // Label tag with provenance
      ctx.fillStyle = isSelected ? "#38bdf8" : strokeColor;
      ctx.font = "bold 15px sans-serif";
      let provText = "";
      if (obj.provenance === "ai_accepted") provText = " [AI]";
      else if (obj.provenance === "human_corrected") provText = " [MOD]";
      else if (obj.provenance === "human_added") provText = " [HUMAN]";
      else if (obj.provenance === "independent_human") provText = " [IND]";

      const tag = isUnassigned ? `ID:${obj.track_id} [CLASS REQUIRED]` : `ID:${obj.track_id} [${obj.category}]${provText}`;
      const textWidth = ctx.measureText(tag).width;
      ctx.fillRect(b.x1, Math.max(0, b.y1 - 24), textWidth + 8, 24);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(tag, b.x1 + 4, Math.max(18, b.y1 - 6));
    });
  }
}

function setupCanvasEvents() {
  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // 1. Check if clicked inside an existing annotated object
    let clickedIdx = -1;
    if (currentFrameData && currentFrameData.objects) {
      for (let i = currentFrameData.objects.length - 1; i >= 0; i--) {
        const b = currentFrameData.objects[i].bbox;
        if (clickX >= b.x1 && clickX <= b.x2 && clickY >= b.y1 && clickY <= b.y2) {
          clickedIdx = i;
          break;
        }
      }
    }

    if (clickedIdx !== -1) {
      selectedObjIndex = clickedIdx;
      updateSidebar();
      redraw();
      return;
    }

    // 2. Check if clicked inside an unaccepted AI proposal (in AI mode)
    if (currentMode === "ai_assisted" && showProposalsLayer && currentProposals) {
      for (let pIdx = 0; pIdx < currentProposals.length; pIdx++) {
        const pb = currentProposals[pIdx].bbox;
        if (clickX >= pb.x1 && clickX <= pb.x2 && clickY >= pb.y1 && clickY <= pb.y2) {
          const match = findMatchingAnnotation(pb);
          if (!match) {
            // Clicked proposal: prompt to accept
            if (confirm(`Accept AI Proposal '${currentProposals[pIdx].category}' as an annotation?`)) {
              acceptProposal(pIdx);
              return;
            }
          }
        }
      }
    }

    // 3. Otherwise start drawing a new bounding box
    isDrawing = true;
    startX = clickX;
    startY = clickY;
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const curX = (e.clientX - rect.left) * scaleX;
    const curY = (e.clientY - rect.top) * scaleY;

    redraw();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(
      Math.min(startX, curX),
      Math.min(startY, curY),
      Math.abs(curX - startX),
      Math.abs(curY - startY)
    );
    ctx.setLineDash([]);
  });

  canvas.addEventListener("mouseup", (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;

    const x1 = Math.max(0, Math.min(startX, endX));
    const y1 = Math.max(0, Math.min(startY, endY));
    const x2 = Math.min(canvas.width, Math.max(startX, endX));
    const y2 = Math.min(canvas.height, Math.max(startY, endY));

    if ((x2 - x1) >= 5 && (y2 - y1) >= 5) {
      const nextTrackId = getNextAvailableTrackId();
      const prov = (currentMode === "independent" ? "independent_human" : "human_added");
      const newObj = {
        object_id: `obj_${Date.now()}`,
        track_id: nextTrackId,
        category: "", // Unassigned: reviewer must explicitly select class
        bbox: { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) },
        occlusion: "none",
        visibility: "clear",
        is_ambiguous: false,
        confirmation_status: "pending_review",
        review_notes: "",
        provenance: prov
      };
      if (!currentFrameData.objects) currentFrameData.objects = [];
      currentFrameData.objects.push(newObj);
      selectedObjIndex = currentFrameData.objects.length - 1;
      updateSidebar();
      renderProposalsList();
      redraw();
    }
  });
}

function getNextAvailableTrackId() {
  if (!currentFrameData || !currentFrameData.objects || currentFrameData.objects.length === 0) return 1;
  const existing = currentFrameData.objects.map(o => o.track_id);
  return Math.max(...existing) + 1;
}

function setupObjectFormEvents() {
  document.getElementById("objTrackId").addEventListener("change", (e) => {
    if (selectedObjIndex >= 0) {
      currentFrameData.objects[selectedObjIndex].track_id = parseInt(e.target.value) || 0;
      markObjectModified(selectedObjIndex);
      updateSidebar();
      redraw();
    }
  });
  document.getElementById("objCategory").addEventListener("change", (e) => {
    if (selectedObjIndex >= 0) {
      currentFrameData.objects[selectedObjIndex].category = e.target.value;
      markObjectModified(selectedObjIndex);
      updateSidebar();
      redraw();
    }
  });
  document.getElementById("objOcclusion").addEventListener("change", (e) => {
    if (selectedObjIndex >= 0) {
      currentFrameData.objects[selectedObjIndex].occlusion = e.target.value;
      markObjectModified(selectedObjIndex);
    }
  });
  document.getElementById("objVisibility").addEventListener("change", (e) => {
    if (selectedObjIndex >= 0) {
      currentFrameData.objects[selectedObjIndex].visibility = e.target.value;
      markObjectModified(selectedObjIndex);
    }
  });
  document.getElementById("objAmbiguous").addEventListener("change", (e) => {
    if (selectedObjIndex >= 0) {
      currentFrameData.objects[selectedObjIndex].is_ambiguous = e.target.checked;
      markObjectModified(selectedObjIndex);
    }
  });
  document.getElementById("deleteObjBtn").addEventListener("click", deleteSelectedObject);

  document.getElementById("confirmFrameBtn").addEventListener("click", () => {
    const revId = document.getElementById("reviewerId").value.trim();
    if (!revId) {
      alert("Please enter Reviewer Name / ID before confirming.");
      return;
    }
    // Block confirmation if any object in this frame has unassigned class
    if (currentFrameData.objects) {
      const unassigned = currentFrameData.objects.find(o => !o.category || o.category.trim() === "");
      if (unassigned) {
        alert(`Cannot confirm frame: Object ID ${unassigned.track_id} has no class assigned. Please select an object class from the dropdown.`);
        return;
      }
    }
    currentFrameData.reviewer_id = revId;
    currentFrameData.confirmed_at_utc = new Date().toISOString();
    currentFrameData.objects.forEach(o => {
      o.confirmation_status = "human_confirmed";
    });
    updateProgressBadge();
    saveDataset(false);
    alert(`Frame ${currentTick} confirmed by ${revId}`);
  });

  document.getElementById("saveDatasetBtn").addEventListener("click", () => saveDataset(true));
}

function markObjectModified(idx) {
  if (idx >= 0 && currentFrameData.objects && currentFrameData.objects[idx]) {
    const obj = currentFrameData.objects[idx];
    if (obj.provenance === "ai_accepted") {
      obj.provenance = "human_corrected";
    }
  }
}

function deleteSelectedObject() {
  if (selectedObjIndex >= 0 && currentFrameData.objects) {
    currentFrameData.objects.splice(selectedObjIndex, 1);
    selectedObjIndex = -1;
    updateSidebar();
    renderProposalsList();
    redraw();
  }
}

function updateSidebar() {
  const form = document.getElementById("objectForm");
  const notice = document.getElementById("noObjectNotice");
  const objList = document.getElementById("objectList");
  const objCount = document.getElementById("objCount");

  const objects = (currentFrameData && currentFrameData.objects) ? currentFrameData.objects : [];
  objCount.textContent = objects.length;
  objList.innerHTML = "";

  objects.forEach((obj, idx) => {
    const div = document.createElement("div");
    const isUnassigned = (!obj.category || obj.category.trim() === "");
    div.className = "object-item" + (idx === selectedObjIndex ? " selected" : "");
    const labelHtml = isUnassigned ? "<span style='color:#f43f5e; font-weight:bold;'>CLASS REQUIRED</span>" : obj.category;

    let provBadge = "";
    if (obj.provenance === "ai_accepted") provBadge = `<span class="provenance-tag prov-ai">AI</span>`;
    else if (obj.provenance === "human_corrected") provBadge = `<span class="provenance-tag prov-corrected">CORRECTED</span>`;
    else if (obj.provenance === "human_added") provBadge = `<span class="provenance-tag prov-human">HUMAN</span>`;
    else if (obj.provenance === "independent_human") provBadge = `<span class="provenance-tag prov-independent">IND</span>`;

    div.innerHTML = `<span><b>ID ${obj.track_id}</b>: ${labelHtml} ${provBadge}</span><span style="color:#aaa;">[${obj.bbox.x1},${obj.bbox.y1}]</span>`;
    div.addEventListener("click", () => {
      selectedObjIndex = idx;
      updateSidebar();
      redraw();
    });
    objList.appendChild(div);
  });

  if (selectedObjIndex >= 0 && selectedObjIndex < objects.length) {
    form.style.display = "block";
    notice.style.display = "none";
    const obj = objects[selectedObjIndex];
    const isUnassigned = (!obj.category || obj.category.trim() === "");
    document.getElementById("objTrackId").value = obj.track_id;
    document.getElementById("objCategory").value = obj.category || "";
    document.getElementById("classRequiredAlert").style.display = isUnassigned ? "block" : "none";
    document.getElementById("objOcclusion").value = obj.occlusion || "none";
    document.getElementById("objVisibility").value = obj.visibility || "clear";
    document.getElementById("objAmbiguous").checked = !!obj.is_ambiguous;

    const provBadge = document.getElementById("objProvenanceBadge");
    const prov = obj.provenance || "draft";
    provBadge.textContent = prov.replace("_", " ").toUpperCase();
    provBadge.className = "provenance-tag " + (prov === "ai_accepted" ? "prov-ai" : (prov === "human_corrected" ? "prov-corrected" : (prov === "independent_human" ? "prov-independent" : "prov-human")));
  } else {
    form.style.display = "none";
    notice.style.display = "block";
  }
}

async function saveDataset(showAlert = true) {
  // Validate that all objects have an assigned category before persisting
  for (const f of dataset.frames) {
    if (f.objects) {
      for (const o of f.objects) {
        if (!o.category || o.category.trim() === "") {
          alert(`Cannot save annotations: Object ID ${o.track_id} in ${f.scenario_id} (Tick ${f.tick}) has no class assigned. Every bounding box must have an assigned class before saving to disk.`);
          return;
        }
      }
    }
  }

  dataset.annotation_mode = currentMode;

  const resp = await fetch("/api/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(dataset),
  });
  const res = await resp.json();
  if (res.status === "ok") {
    document.getElementById("datasetHash").textContent = res.sha256.substring(0, 16) + "...";
    if (showAlert) alert("Annotations successfully saved and hashed.");
  } else {
    alert("Save error: " + res.message);
  }
}

window.onload = init;
</script>
</body>
</html>
"""


def create_app(
    annotations_path: Path,
    frames_dir: Path,
    predictions_path: Optional[Path] = None,
    default_mode: str = "ai_assisted",
    enable_live_detector: bool = True,
) -> Flask:
    """Create Flask annotation server app with AI proposal support."""
    annotations_path = Path(annotations_path).resolve()
    frames_dir = Path(frames_dir).resolve()

    # Auto-detect default predictions cache if not specified
    if predictions_path is None:
        default_pred_file = Path("reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json")
        if default_pred_file.exists():
            predictions_path = default_pred_file.resolve()
    elif predictions_path is not None:
        predictions_path = Path(predictions_path).resolve()

    app = Flask(__name__)

    # Lazy detector holder
    live_detector_holder: List[Any] = [None]

    def _get_detector():
        if live_detector_holder[0] is None and enable_live_detector:
            try:
                from cognix.evaluation.torchvision_detector import TorchVisionObjectDetector
                live_detector_holder[0] = TorchVisionObjectDetector(confidence_threshold=0.3, device="cpu")
            except Exception:
                live_detector_holder[0] = False
        return live_detector_holder[0] if live_detector_holder[0] is not False else None

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/annotations", methods=["GET"])
    def get_annotations():
        if annotations_path.exists():
            dataset = AnnotationDataset.load_json(annotations_path)
            if not dataset.annotation_mode:
                dataset.annotation_mode = default_mode
        else:
            dataset = AnnotationDataset(
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                annotation_mode=default_mode,
            )
        return jsonify(dataset.to_dict())

    @app.route("/api/annotations", methods=["POST"])
    def save_annotations():
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON payload"}), 400
        try:
            dataset = AnnotationDataset.from_dict(data)
            dataset.validate()
            sha256 = dataset.save_json(annotations_path)
            return jsonify({"status": "ok", "sha256": sha256})
        except (ValidationError, ValueError) as e:
            return jsonify({"status": "error", "message": str(e)}), 422
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/proposals", methods=["GET"])
    def get_proposals():
        scenario = request.args.get("scenario", "")
        tick = request.args.get("tick", "0")
        try:
            tick_int = int(tick)
        except ValueError:
            return jsonify({"error": f"Invalid tick: {tick}"}), 400

        sc_dir = frames_dir / scenario.replace("/", "_")
        img_path = (sc_dir / f"{tick_int:06d}.jpg").resolve()
        if not img_path.exists():
            flat_path = (frames_dir / f"{tick_int:06d}.jpg").resolve()
            if flat_path.exists():
                img_path = flat_path
            else:
                return jsonify({"error": f"Image not found at {img_path.name}"}), 404

        img_bytes = img_path.read_bytes()
        img_sha256 = hashlib.sha256(img_bytes).hexdigest()

        proposals = []
        source = "live_torchvision"

        # 1. Attempt cached predictions lookup if available
        if predictions_path and Path(predictions_path).exists():
            try:
                with open(predictions_path, "r", encoding="utf-8") as pf:
                    cached_data = json.load(pf)

                key = f"{scenario}:{tick_int}"
                m7_tracks = cached_data.get("m7_active_tracks_by_frame", {}).get(key)
                if m7_tracks is not None:
                    source = "cached_m7_tracks"
                    for t in m7_tracks:
                        lbox = t.get("latest_box", {})
                        raw_class = t.get("class_name", lbox.get("class_name", "unknown"))
                        tax_class = map_detector_class_to_taxonomy(raw_class)
                        if tax_class != UNSUPPORTED_CLASS_CATEGORY:
                            proposals.append({
                                "proposal_id": f"prop_m7_{t.get('track_id')}_{tick_int}",
                                "track_id": int(t.get("track_id", 0)),
                                "category": tax_class,
                                "raw_detector_class": raw_class,
                                "score": round(float(lbox.get("score", 1.0)), 4),
                                "bbox": {
                                    "x1": round(float(lbox.get("x1", 0.0)), 1),
                                    "y1": round(float(lbox.get("y1", 0.0)), 1),
                                    "x2": round(float(lbox.get("x2", 0.0)), 1),
                                    "y2": round(float(lbox.get("y2", 0.0)), 1),
                                },
                            })
                elif key in cached_data.get("m6_detections_by_frame", {}):
                    source = "cached_m6_detections"
                    m6_dets = cached_data["m6_detections_by_frame"][key]
                    for idx, det in enumerate(m6_dets):
                        raw_class = det.get("class_name", "unknown")
                        tax_class = map_detector_class_to_taxonomy(raw_class)
                        if tax_class != UNSUPPORTED_CLASS_CATEGORY:
                            proposals.append({
                                "proposal_id": f"prop_m6_{idx}_{tick_int}",
                                "track_id": idx + 1,
                                "category": tax_class,
                                "raw_detector_class": raw_class,
                                "score": round(float(det.get("score", 1.0)), 4),
                                "bbox": {
                                    "x1": round(float(det.get("x1", 0.0)), 1),
                                    "y1": round(float(det.get("y1", 0.0)), 1),
                                    "x2": round(float(det.get("x2", 0.0)), 1),
                                    "y2": round(float(det.get("y2", 0.0)), 1),
                                },
                            })
            except Exception:
                pass

        # 2. Fallback to live detector if cached predictions unavailable for this frame
        if not proposals and enable_live_detector:
            detector = _get_detector()
            if detector is not None:
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    frame_dets = detector.detect(pil_img, scenario_id=scenario, tick=tick_int)
                    source = "live_torchvision"
                    for idx, det in enumerate(frame_dets.boxes):
                        tax_class = map_detector_class_to_taxonomy(det.class_name)
                        if tax_class != UNSUPPORTED_CLASS_CATEGORY:
                            proposals.append({
                                "proposal_id": f"prop_live_{idx}_{tick_int}",
                                "track_id": idx + 1,
                                "category": tax_class,
                                "raw_detector_class": det.class_name,
                                "score": round(float(det.score), 4),
                                "bbox": {
                                    "x1": round(float(det.x1), 1),
                                    "y1": round(float(det.y1), 1),
                                    "x2": round(float(det.x2), 1),
                                    "y2": round(float(det.y2), 1),
                                },
                            })
                except Exception:
                    pass

        return jsonify({
            "scenario": scenario,
            "tick": tick_int,
            "image_hash_sha256": img_sha256,
            "source": source,
            "num_proposals": len(proposals),
            "proposals": proposals,
            "known_limitations": "Faster R-CNN MobileNetV3 (COCO pretrained) does not detect custom CARLA roadway crates/props.",
        })

    @app.route("/api/frame_image")
    def get_frame_image():
        scenario = request.args.get("scenario", "")
        tick = request.args.get("tick", "0")
        try:
            tick_int = int(tick)
        except ValueError:
            return jsonify({"error": f"Invalid tick: {tick}"}), 400

        sc_dir = frames_dir / scenario.replace("/", "_")
        img_path = (sc_dir / f"{tick_int:06d}.jpg").resolve()
        if not img_path.exists():
            flat_path = (frames_dir / f"{tick_int:06d}.jpg").resolve()
            if flat_path.exists():
                img_path = flat_path
            else:
                return jsonify({"error": f"Image not found at {img_path.name}"}), 404
        return send_file(img_path, mimetype="image/jpeg")

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Cognix M8 Annotation Studio Server")
    parser.add_argument("--port", type=int, default=5050, help="Port to listen on")
    parser.add_argument(
        "--annotations",
        type=str,
        default="reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json",
        help="Optional path to cached model predictions JSON",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ai_assisted",
        choices=["ai_assisted", "independent"],
        help="Default annotation workflow mode ('ai_assisted' or 'independent')",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions) if args.predictions else None
    app = create_app(Path(args.annotations), Path(args.frames_dir), predictions_path=pred_path, default_mode=args.mode)
    print(f"Starting Cognix M8 Annotation Studio in {args.mode.upper()} mode at http://127.0.0.1:{args.port}/")
    app.run(host="127.0.0.1", port=args.port, debug=False)
