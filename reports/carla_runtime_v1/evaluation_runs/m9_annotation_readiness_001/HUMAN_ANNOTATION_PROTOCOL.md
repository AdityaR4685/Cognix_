# Cognix Human Annotation & Review Protocol (Version 1.0)

**Applicability:** CARLA 2D Perception & Object Tracking Evaluation
**Target Resolution:** $1920 \times 1080$ RGB Full HD
**Sampling Stride:** 5 Simulation Ticks (e.g. 0, 5, 10, 15, ..., 45)
**Document Location:** `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/HUMAN_ANNOTATION_PROTOCOL.md`

---

## 1. Core Principles & Scientific Honesty

1. **Human Authority:** Ground-truth annotations must represent the independent judgment of an identified human reviewer. AI assistants and automated algorithms **must never** assign themselves reviewer identities or mark automated predictions as human-approved.
2. **Unbiased Visual Inspection:** Reviewers must initially inspect RGB camera frames clean of detector bounding boxes, confidence labels, or anomaly segmentation masks to prevent anchoring bias.
3. **Conservative Labeling:** Never guess an identity or boundary that the visual evidence cannot support. When lighting, heavy rain, or occlusion obscures an object's boundary, use the documented ambiguity rules.

---

## 2. Object Categories

| Category Key | Display Name | Definition and Boundaries | Inclusions / Exclusions |
| :--- | :--- | :--- | :--- |
| `person` | **Person / Pedestrian** | Any human figure (walking, running, standing, jaywalking, or on sidewalk). | Includes carried backpacks/handbags. Excludes statues, posters, or distant silhouettes (< 10 px tall). |
| `vehicle` | **Vehicle** | Motorized road vehicles: passenger cars, SUVs, vans, pickup trucks, heavy trucks, buses. | Enclose the full visible chassis, wheels, and side mirrors. Excludes ego car hood (see Corridor rules). |
| `bicycle_or_motorcycle` | **Bike / Motorcycle** | Two-wheeled vehicles, including cyclist or rider. | Box must enclose **both** the rider and the cycle as a single dynamic unit. |
| `road_debris_or_prop` | **Road Debris / Prop** | Custom roadway obstacles: wooden cargo crates, construction cones, trash cans, fallen cargo. | Objects obstructing or adjacent to the drivable lane. Excludes permanent roadside buildings/poles. |
| `ignore_ambiguous` | **Ignore / Ambiguous** | Regions where image artifacts, severe reflections, extreme distance, or clutter prevent a reliable label. | Prevents false-alarm penalization in downstream evaluator metrics. |

---

## 3. Bounding Box Geometry & Coordinate Convention

* **Coordinate Format:** Standard image-space 2D coordinates `xyxy`:
  $$\text{bbox} = [x_1, y_1, x_2, y_2]$$
  where $(x_1, y_1)$ is the top-left corner and $(x_2, y_2)$ is the bottom-right corner.
* **Image Space Bounds:** Origin $(0, 0)$ is top-left.
  $$0.0 \le x_1 < x_2 \le 1920.0, \quad 0.0 \le y_1 < y_2 \le 1080.0$$
* **Tight Boundary Rule:** Boxes must enclose the visible boundary of the object tightly. Do not pad excess background.
* **Truncated Objects (Image Edge):** If an object extends beyond the camera field of view, clip the box cleanly at the image boundary ($x=0$, $x=1920$, $y=0$, or $y=1080$). Set `occlusion="partial"` or `"heavy"` to note edge truncation.

---

## 4. Occlusion, Visibility, and Ambiguity Standards

### A. Occlusion States
* `none`: 90–100% of the object body/outline is directly visible.
* `partial`: 50–90% visible (e.g. lower legs hidden behind guardrail, vehicle rear partially blocked by another car).
* `heavy`: Less than 50% visible (e.g. only head/shoulders visible, or vehicle mostly occluded by roadside building).

### B. Visibility States
* `clear`: Normal daylight, clear atmospheric conditions, distinct edges.
* `degraded_weather`: Heavy rain, lens droplets, water spray, or fog softening object contours.
* `glare_or_shadow`: Severe headlight flare, wet asphalt glare, or pitch-black shadow regions.

### C. When to Mark Ambiguous (`is_ambiguous = true`)
* Mark an object ambiguous when you can perceive that an object exists, but cannot determine its exact category or boundary due to motion blur or low contrast.
* If a region contains visual artifacts that might trigger false alarms (e.g. reflections resembling a pedestrian), draw an `ignore_ambiguous` box around it.

---

## 5. Temporal Track Identity (`track_id`) Continuity

1. **Stable Integer Identity:** Assign a consistent integer `track_id` (e.g. `1`, `2`, `3`) to the same physical object across consecutive frames within a scenario.
2. **Sampling Stride Awareness (5 Simulation Ticks):**
   * The evaluation samples every 5th tick (e.g. tick 0, 5, 10, 15...).
   * Objects will exhibit discrete pixel jumps between sampled frames ($\sim 5\times$ single-tick displacement).
   * Do not expect smooth single-frame optical flow. Validate track identity by checking consistent object appearance, direction of travel, and lane position.
3. **Identity Preservation During Dropout / Occlusion:**
   * If an object is temporarily occluded behind a post or car for 1 or 2 sampled frames and reappears, maintain the **same `track_id`**.
   * If an object leaves the field of view completely and a new, distinct object enters later, assign a **new `track_id`**. Never recycle IDs.
4. **Scenario Boundary Strictness:**
   * Track IDs are strictly local to a scenario sequence.
   * On scenario transitions (e.g. moving from `running-pedestrian/scenario-1` to `spawn-props/scenario-1`), the tracker and annotation IDs reset completely.

---

## 6. Review Confirmation & Attribution Lifecycle

* **Confirmation States:**
  1. `unconfirmed_draft`: Automated candidate proposal or initial unverified sketch.
  2. `pending_review`: Formal item in the review queue awaiting human sign-off.
  3. `human_confirmed`: Explicitly validated and signed by a named human auditor.
* **Reviewer Attribution Requirements:**
  - `reviewer_id`: Full name or identifier of the reviewer (e.g. `"Aditya (Safety Auditor)"`).
  - `confirmed_at_utc`: ISO 8601 UTC timestamp generated at time of confirmation.
  - An annotation **cannot** be saved as `human_confirmed` if `reviewer_id` is empty.
* **Reviewed-Empty Frames:**
  - If a frame has zero objects in view (clear roadway), the reviewer leaves the object list empty, enters their `reviewer_id`, and clicks **Confirm & Sign Frame**.
  - This marks the frame as legitimately reviewed with 0 objects, rather than skipped or uninspected.

---

## 7. Review Studio Launch & Workflow Instructions

To launch the local annotation studio targeting the working copy:

```powershell
# 1. Activate dedicated perception environment
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5050 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames
```

### Reviewer Step-by-Step Checklist:
1. Open Chrome/Edge at `http://127.0.0.1:5050/`.
2. Inspect the displayed RGB image for pedestrians, vehicles, bikes, and roadway debris.
3. Click and drag to create boxes around all valid objects.
4. Set category, track ID, occlusion, and visibility in the right sidebar.
5. Enter your name in `Reviewer Name / ID`.
6. Click **Confirm & Sign Frame** (or press shortcut).
7. Advance to the next frame (`D` or `Right Arrow`).
8. Repeat across all 10 ticks for each of the 4 scenarios.
9. Click **Save Annotations to Disk** to update the signed working copy.
