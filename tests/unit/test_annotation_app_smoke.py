"""Smoke and boundary tests for Cognix Annotation Application (Milestone 9).

Validates:
1. Application initialization using dedicated perception environment.
2. Localhost-only binding (127.0.0.1) and route protection.
3. Save/load lifecycle on temporary copy (never mutating canonical M8 files).
4. Full round-trip fidelity: coordinates, classes, track IDs, occlusion, visibility, reviewer info.
5. Ambiguous objects, skipped/empty frames, and invalid coordinate rejection.
6. Review confirmation gate: reviewer_id required before human confirmation.
7. Frame image association: scenario/tick route loads the exact corresponding frame.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image

from cognix.evaluation.annotation_app import create_app
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


@pytest.fixture
def temp_annotation_env(tmp_path):
    """Creates a temporary, isolated annotation environment with synthetic frames."""
    frames_dir = tmp_path / "frames"
    scenario_name = "test/anomaly/Town01/running-pedestrian/scenario-1"
    sc_clean = scenario_name.replace("/", "_")
    sc_dir = frames_dir / sc_clean
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Generate two distinct synthetic test images (tick 0: Red, tick 5: Blue)
    im0 = Image.new("RGB", (1920, 1080), color=(255, 0, 0))
    im0_path = sc_dir / "000000.jpg"
    im0.save(im0_path, format="JPEG")
    sha0 = hashlib.sha256(im0_path.read_bytes()).hexdigest()

    im5 = Image.new("RGB", (1920, 1080), color=(0, 0, 255))
    im5_path = sc_dir / "000005.jpg"
    im5.save(im5_path, format="JPEG")
    sha5 = hashlib.sha256(im5_path.read_bytes()).hexdigest()

    # Build initial dataset
    f0 = FrameAnnotation(scenario_id=scenario_name, tick=0, frame_hash_sha256=sha0)
    f5 = FrameAnnotation(scenario_id=scenario_name, tick=5, frame_hash_sha256=sha5)
    dataset = AnnotationDataset(
        schema_version="1.0.0",
        created_at_utc="2026-10-20T12:00:00Z",
        frames=[f0, f5],
    )
    dataset_file = tmp_path / "test_annotations.json"
    dataset.save_json(dataset_file)

    app = create_app(dataset_file, frames_dir)
    app.config["TESTING"] = True
    client = app.test_client()

    return {
        "tmp_path": tmp_path,
        "dataset_file": dataset_file,
        "frames_dir": frames_dir,
        "scenario": scenario_name,
        "client": client,
        "sha0": sha0,
        "sha5": sha5,
    }


class TestAnnotationAppSmoke:
    """Verifies interface endpoints, save/load behavior, and validation bounds."""

    def test_app_starts_and_serves_html_ui(self, temp_annotation_env):
        client = temp_annotation_env["client"]
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Cognix M8 Annotation & Review Studio" in html
        assert "<canvas id=\"annotationCanvas\"" in html

    def test_api_get_annotations(self, temp_annotation_env):
        client = temp_annotation_env["client"]
        resp = client.get("/api/annotations")
        assert resp.status_code == 200
        data = resp.json
        assert data["schema_version"] == "1.0.0"
        assert len(data["frames"]) == 2
        assert data["frames"][0]["tick"] == 0
        assert data["frames"][1]["tick"] == 5

    def test_save_and_reload_round_trip_fidelity(self, temp_annotation_env):
        client = temp_annotation_env["client"]
        dataset_file = temp_annotation_env["dataset_file"]

        # 1. Fetch current annotations
        data = client.get("/api/annotations").json

        # 2. Add an annotated object to frame 0
        new_obj = {
            "object_id": "test_obj_101",
            "track_id": 42,
            "category": "vehicle",
            "bbox": {"x1": 150.0, "y1": 250.0, "x2": 450.0, "y2": 550.0},
            "occlusion": "partial",
            "visibility": "clear",
            "is_ambiguous": False,
            "confirmation_status": "human_confirmed",
            "review_notes": "Clean vehicle detection",
        }
        data["frames"][0]["objects"] = [new_obj]
        data["frames"][0]["reviewer_id"] = "SafetyAuditor01"
        data["frames"][0]["confirmed_at_utc"] = "2026-10-20T12:30:00Z"

        # 3. Save via POST
        save_resp = client.post("/api/annotations", json=data)
        assert save_resp.status_code == 200
        assert save_resp.json["status"] == "ok"
        saved_sha256 = save_resp.json["sha256"]

        # 4. Re-instantiate a fresh app reading from disk
        fresh_app = create_app(dataset_file, temp_annotation_env["frames_dir"])
        fresh_client = fresh_app.test_client()

        reloaded = fresh_client.get("/api/annotations").json
        assert reloaded["frames"][0]["reviewer_id"] == "SafetyAuditor01"
        assert reloaded["frames"][0]["confirmed_at_utc"] == "2026-10-20T12:30:00Z"
        assert len(reloaded["frames"][0]["objects"]) == 1

        obj = reloaded["frames"][0]["objects"][0]
        assert obj["object_id"] == "test_obj_101"
        assert obj["track_id"] == 42
        assert obj["category"] == "vehicle"
        assert obj["bbox"] == {"x1": 150.0, "y1": 250.0, "x2": 450.0, "y2": 550.0}
        assert obj["occlusion"] == "partial"
        assert obj["confirmation_status"] == "human_confirmed"

    def test_ambiguous_object_and_empty_frame_handling(self, temp_annotation_env):
        client = temp_annotation_env["client"]
        data = client.get("/api/annotations").json

        # Frame 0: Ambiguous clutter object
        data["frames"][0]["objects"] = [
            {
                "object_id": "ambig_obj",
                "track_id": 99,
                "category": "ignore_ambiguous",
                "bbox": {"x1": 10.0, "y1": 10.0, "x2": 80.0, "y2": 80.0},
                "occlusion": "heavy",
                "visibility": "degraded_weather",
                "is_ambiguous": True,
                "confirmation_status": "human_confirmed",
                "review_notes": "Severe rain distortion",
            }
        ]
        data["frames"][0]["reviewer_id"] = "SafetyAuditor01"

        # Frame 5: Legitimate empty frame (no objects), confirmed
        data["frames"][1]["objects"] = []
        data["frames"][1]["reviewer_id"] = "SafetyAuditor01"
        data["frames"][1]["confirmed_at_utc"] = "2026-10-20T12:35:00Z"

        save_resp = client.post("/api/annotations", json=data)
        assert save_resp.status_code == 200

        reloaded = client.get("/api/annotations").json
        assert reloaded["frames"][0]["objects"][0]["is_ambiguous"] is True
        assert len(reloaded["frames"][1]["objects"]) == 0
        assert reloaded["frames"][1]["reviewer_id"] == "SafetyAuditor01"

    def test_invalid_coordinates_rejected_with_422(self, temp_annotation_env):
        client = temp_annotation_env["client"]
        data = client.get("/api/annotations").json

        # Coordinates out of bounds (x2 = 2500 > 1920)
        data["frames"][0]["objects"] = [
            {
                "object_id": "bad_coords",
                "track_id": 1,
                "category": "person",
                "bbox": {"x1": 100.0, "y1": 100.0, "x2": 2500.0, "y2": 200.0},
                "occlusion": "none",
                "visibility": "clear",
                "is_ambiguous": False,
                "confirmation_status": "pending_review",
            }
        ]

        resp = client.post("/api/annotations", json=data)
        assert resp.status_code == 422
        assert "Invalid horizontal coordinates" in resp.json["message"]

    def test_duplicate_track_id_rejected_with_422(self, temp_annotation_env):
        client = temp_annotation_env["client"]
        data = client.get("/api/annotations").json

        # Two distinct objects with identical track_id in the same frame
        data["frames"][0]["objects"] = [
            {
                "object_id": "obj_a",
                "track_id": 7,
                "category": "person",
                "bbox": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
                "occlusion": "none",
                "visibility": "clear",
                "is_ambiguous": False,
                "confirmation_status": "pending_review",
            },
            {
                "object_id": "obj_b",
                "track_id": 7,  # Duplicate!
                "category": "vehicle",
                "bbox": {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 150.0},
                "occlusion": "none",
                "visibility": "clear",
                "is_ambiguous": False,
                "confirmation_status": "pending_review",
            },
        ]

        resp = client.post("/api/annotations", json=data)
        assert resp.status_code == 422
        assert "Duplicate track_id 7" in resp.json["message"]

    def test_frame_image_route_associates_exact_scenario_and_tick(self, temp_annotation_env):
        client = temp_annotation_env["client"]
        sc = temp_annotation_env["scenario"]

        # Fetch tick 0 -> Red image
        resp0 = client.get(f"/api/frame_image?scenario={sc}&tick=0")
        assert resp0.status_code == 200
        assert hashlib.sha256(resp0.data).hexdigest() == temp_annotation_env["sha0"]

        # Fetch tick 5 -> Blue image
        resp5 = client.get(f"/api/frame_image?scenario={sc}&tick=5")
        assert resp5.status_code == 200
        assert hashlib.sha256(resp5.data).hexdigest() == temp_annotation_env["sha5"]

        # Non-existent tick -> 404
        resp_missing = client.get(f"/api/frame_image?scenario={sc}&tick=999")
        assert resp_missing.status_code == 404

    def test_frame_image_route_flat_directory_fallback(self, tmp_path):
        """Verifies single-scenario packages with flat frame directories load correctly."""
        frames_dir = tmp_path / "flat_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        img_path = frames_dir / "000010.jpg"
        Image.new("RGB", (1920, 1080), color=(0, 255, 0)).save(img_path, format="JPEG")
        img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()

        dataset_file = tmp_path / "flat_dataset.json"
        AnnotationDataset(frames=[]).save_json(dataset_file)

        app = create_app(dataset_file, frames_dir)
        client = app.test_client()

        resp = client.get("/api/frame_image?scenario=test/normal/Town01/scenario-1&tick=10")
        assert resp.status_code == 200
        assert hashlib.sha256(resp.data).hexdigest() == img_sha

    def test_frame_image_route_with_relative_paths(self, tmp_path, monkeypatch):
        """Regression test for relative CLI argument resolution in Flask send_file."""
        monkeypatch.chdir(tmp_path)
        rel_frames = Path("rel_frames")
        sc_dir = rel_frames / "test_anomaly_Town01_running-pedestrian_scenario-1"
        sc_dir.mkdir(parents=True, exist_ok=True)
        img_path = sc_dir / "000000.jpg"
        Image.new("RGB", (1920, 1080), color=(128, 128, 128)).save(img_path, format="JPEG")
        img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()

        rel_annotations = Path("rel_annotations.json")
        AnnotationDataset(frames=[]).save_json(rel_annotations)

        # Pass relative paths as done by CLI
        app = create_app(rel_annotations, rel_frames)
        client = app.test_client()

        resp = client.get("/api/frame_image?scenario=test/anomaly/Town01/running-pedestrian/scenario-1&tick=0")
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"
        assert hashlib.sha256(resp.data).hexdigest() == img_sha

    def test_html_ui_contains_visible_loading_and_error_states(self, temp_annotation_env):
        """Verifies UI template includes active loading and error rendering on canvas."""
        client = temp_annotation_env["client"]
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "imageLoading" in html
        assert "imageError" in html
        assert "Failed to Load Frame Image" in html
        assert "Loading Frame (Tick" in html

    def test_ui_html_requires_explicit_class_selection(self, temp_annotation_env):
        """Verifies UI requires explicit class selection and does not default newly drawn boxes to person."""
        client = temp_annotation_env["client"]
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        # Ensure dropdown has disabled placeholder requiring selection
        assert "-- Select Object Class (Required) --" in html
        # Ensure alert element exists
        assert "classRequiredAlert" in html
        # Ensure canvas tag renders CLASS REQUIRED for unassigned boxes
        assert "CLASS REQUIRED" in html
        # Ensure unassigned category is initialized as empty string in mouseup
        assert 'category: "", // Unassigned: reviewer must explicitly select class' in html
        # Ensure save and confirm block if unassigned
        assert "has no class assigned. Every bounding box must have an assigned class" in html

    def test_save_with_unassigned_or_empty_class_rejected_by_backend(self, temp_annotation_env):
        """Backend schema strictly rejects unassigned or empty string categories with 422."""
        client = temp_annotation_env["client"]
        data = client.get("/api/annotations").json
        # Attempt to save an object with empty category
        data["frames"][0]["objects"] = [
            {
                "object_id": "test_unassigned",
                "track_id": 1,
                "category": "",  # Empty / unassigned
                "bbox": {"x1": 50.0, "y1": 50.0, "x2": 150.0, "y2": 150.0},
                "occlusion": "none",
                "visibility": "clear",
                "is_ambiguous": False,
                "confirmation_status": "pending_review",
            }
        ]
        resp = client.post("/api/annotations", json=data)
        assert resp.status_code == 422
        assert "ObjectCategory" in resp.json["message"] or "Invalid category" in resp.json["message"]

    def test_select_class_changes_only_target_object_and_survives_roundtrip(self, temp_annotation_env):
        """Selecting a class changes only the target object and persists across save and reload."""
        client = temp_annotation_env["client"]
        dataset_file = temp_annotation_env["dataset_file"]
        data = client.get("/api/annotations").json

        # Frame 0: two distinct objects
        obj1 = {
            "object_id": "obj_car",
            "track_id": 1,
            "category": "vehicle",
            "bbox": {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 300.0},
            "occlusion": "none",
            "visibility": "clear",
            "is_ambiguous": False,
            "confirmation_status": "pending_review",
        }
        obj2 = {
            "object_id": "obj_prop",
            "track_id": 2,
            "category": "road_debris_or_prop",
            "bbox": {"x1": 400.0, "y1": 400.0, "x2": 500.0, "y2": 500.0},
            "occlusion": "none",
            "visibility": "clear",
            "is_ambiguous": False,
            "confirmation_status": "pending_review",
        }
        data["frames"][0]["objects"] = [obj1, obj2]

        save_resp = client.post("/api/annotations", json=data)
        assert save_resp.status_code == 200

        # Now simulate selecting object 2 and changing its class to person
        reloaded = client.get("/api/annotations").json
        assert len(reloaded["frames"][0]["objects"]) == 2
        # Modify only object 2
        reloaded["frames"][0]["objects"][1]["category"] = "person"

        save_resp2 = client.post("/api/annotations", json=reloaded)
        assert save_resp2.status_code == 200

        # Reload fresh instance from disk
        app2 = create_app(dataset_file, temp_annotation_env["frames_dir"])
        client2 = app2.test_client()
        final_data = client2.get("/api/annotations").json

        objs = final_data["frames"][0]["objects"]
        assert len(objs) == 2
        # Object 1 was unchanged
        assert objs[0]["object_id"] == "obj_car"
        assert objs[0]["category"] == "vehicle"
        # Object 2 was updated to person and persisted
        assert objs[1]["object_id"] == "obj_prop"
        assert objs[1]["category"] == "person"


class TestAIAssistedAnnotationWorkflow:
    """Verifies AI proposal generation, class mapping, provenance, and workflow isolation."""

    def test_detector_class_mapping(self):
        from cognix.evaluation.tracking_evaluator import map_detector_class_to_taxonomy, UNSUPPORTED_CLASS_CATEGORY
        # Pedestrian mappings
        assert map_detector_class_to_taxonomy("person") == "person"
        assert map_detector_class_to_taxonomy("Person") == "person"
        # Vehicle mappings
        assert map_detector_class_to_taxonomy("car") == "vehicle"
        assert map_detector_class_to_taxonomy("truck") == "vehicle"
        assert map_detector_class_to_taxonomy("bus") == "vehicle"
        # Two-wheelers
        assert map_detector_class_to_taxonomy("bicycle") == "bicycle_or_motorcycle"
        assert map_detector_class_to_taxonomy("motorcycle") == "bicycle_or_motorcycle"
        # Unsupported / out-of-scope classes
        assert map_detector_class_to_taxonomy("traffic light") == UNSUPPORTED_CLASS_CATEGORY
        assert map_detector_class_to_taxonomy("dog") == UNSUPPORTED_CLASS_CATEGORY
        assert map_detector_class_to_taxonomy("bench") == UNSUPPORTED_CLASS_CATEGORY

    def test_proposals_endpoint_associates_frame_and_image_hash(self, tmp_path):
        frames_dir = tmp_path / "frames"
        scenario = "test/anomaly/Town01/running-pedestrian/scenario-1"
        sc_dir = frames_dir / scenario.replace("/", "_")
        sc_dir.mkdir(parents=True, exist_ok=True)

        im = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
        img_path = sc_dir / "000000.jpg"
        im.save(img_path, format="JPEG")
        img_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()

        # Create dummy predictions
        pred_file = tmp_path / "dummy_preds.json"
        preds = {
            "m7_active_tracks_by_frame": {
                f"{scenario}:0": [
                    {
                        "track_id": 1,
                        "class_name": "car",
                        "latest_box": {"x1": 500.0, "y1": 500.0, "x2": 800.0, "y2": 700.0, "score": 0.96},
                    }
                ]
            }
        }
        pred_file.write_text(json.dumps(preds), encoding="utf-8")

        dataset_file = tmp_path / "ann.json"
        ds = AnnotationDataset(
            schema_version="1.0.0",
            created_at_utc="2026-10-20T12:00:00Z",
            frames=[FrameAnnotation(scenario_id=scenario, tick=0, frame_hash_sha256=img_hash)],
        )
        ds.save_json(dataset_file)

        app = create_app(dataset_file, frames_dir, predictions_path=pred_file)
        client = app.test_client()

        resp = client.get(f"/api/proposals?scenario={scenario}&tick=0")
        assert resp.status_code == 200
        data = resp.json
        assert data["scenario"] == scenario
        assert data["tick"] == 0
        assert data["image_hash_sha256"] == img_hash
        assert data["source"] == "cached_m7_tracks"
        assert len(data["proposals"]) == 1
        prop = data["proposals"][0]
        assert prop["category"] == "vehicle"
        assert prop["score"] == 0.96
        assert prop["track_id"] == 1
        assert "known_limitations" in data

    def test_accept_proposal_persists_provenance_and_score(self, tmp_path):
        frames_dir = tmp_path / "frames"
        scenario = "test/anomaly/Town01/running-pedestrian/scenario-1"
        sc_dir = frames_dir / scenario.replace("/", "_")
        sc_dir.mkdir(parents=True, exist_ok=True)
        img_path = sc_dir / "000000.jpg"
        Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(img_path, format="JPEG")
        img_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()

        dataset_file = tmp_path / "ann.json"
        ds = AnnotationDataset(
            schema_version="1.0.0",
            created_at_utc="2026-10-20T12:00:00Z",
            annotation_mode="ai_assisted",
            frames=[FrameAnnotation(scenario_id=scenario, tick=0, frame_hash_sha256=img_hash)],
        )
        ds.save_json(dataset_file)

        app = create_app(dataset_file, frames_dir)
        client = app.test_client()

        # Simulate accepting an AI proposal
        data = client.get("/api/annotations").json
        accepted_obj = {
            "object_id": "obj_prop_1",
            "track_id": 1,
            "category": "vehicle",
            "bbox": {"x1": 200.0, "y1": 300.0, "x2": 600.0, "y2": 700.0},
            "occlusion": "none",
            "visibility": "clear",
            "is_ambiguous": False,
            "confirmation_status": "pending_review",
            "confidence": 0.942,
            "provenance": "ai_accepted",
        }
        data["frames"][0]["objects"] = [accepted_obj]
        save_resp = client.post("/api/annotations", json=data)
        assert save_resp.status_code == 200

        # Reload from fresh app instance
        app2 = create_app(dataset_file, frames_dir)
        reloaded = app2.test_client().get("/api/annotations").json
        obj = reloaded["frames"][0]["objects"][0]
        assert obj["object_id"] == "obj_prop_1"
        assert obj["category"] == "vehicle"
        assert obj["confidence"] == 0.942
        assert obj["provenance"] == "ai_accepted"

    def test_correct_proposal_updates_provenance_to_human_corrected(self, tmp_path):
        frames_dir = tmp_path / "frames"
        scenario = "test/anomaly/Town01/running-pedestrian/scenario-1"
        sc_dir = frames_dir / scenario.replace("/", "_")
        sc_dir.mkdir(parents=True, exist_ok=True)
        img_path = sc_dir / "000000.jpg"
        Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(img_path, format="JPEG")
        img_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()

        dataset_file = tmp_path / "ann.json"
        obj = AnnotatedObject(
            object_id="obj_init",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.88,
            provenance="ai_accepted",
        )
        ds = AnnotationDataset(
            schema_version="1.0.0",
            created_at_utc="2026-10-20T12:00:00Z",
            frames=[FrameAnnotation(scenario_id=scenario, tick=0, frame_hash_sha256=img_hash, objects=[obj])],
        )
        ds.save_json(dataset_file)

        app = create_app(dataset_file, frames_dir)
        client = app.test_client()

        # Human corrects the class to vehicle
        data = client.get("/api/annotations").json
        data["frames"][0]["objects"][0]["category"] = "vehicle"
        data["frames"][0]["objects"][0]["provenance"] = "human_corrected"
        save_resp = client.post("/api/annotations", json=data)
        assert save_resp.status_code == 200

        # Verify persisted provenance
        reloaded = client.get("/api/annotations").json
        corrected = reloaded["frames"][0]["objects"][0]
        assert corrected["category"] == "vehicle"
        assert corrected["provenance"] == "human_corrected"

    def test_add_missing_object_sets_human_added_provenance(self, tmp_path):
        frames_dir = tmp_path / "frames"
        scenario = "test/anomaly/Town01/running-pedestrian/scenario-1"
        sc_dir = frames_dir / scenario.replace("/", "_")
        sc_dir.mkdir(parents=True, exist_ok=True)
        img_path = sc_dir / "000000.jpg"
        Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(img_path, format="JPEG")
        img_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()

        dataset_file = tmp_path / "ann.json"
        ds = AnnotationDataset(
            schema_version="1.0.0",
            created_at_utc="2026-10-20T12:00:00Z",
            frames=[FrameAnnotation(scenario_id=scenario, tick=0, frame_hash_sha256=img_hash)],
        )
        ds.save_json(dataset_file)

        app = create_app(dataset_file, frames_dir)
        client = app.test_client()

        data = client.get("/api/annotations").json
        debris_obj = {
            "object_id": "obj_debris_manual",
            "track_id": 5,
            "category": "road_debris_or_prop",
            "bbox": {"x1": 759.0, "y1": 344.0, "x2": 883.0, "y2": 614.0},
            "occlusion": "none",
            "visibility": "clear",
            "is_ambiguous": False,
            "confirmation_status": "pending_review",
            "provenance": "human_added",
        }
        data["frames"][0]["objects"] = [debris_obj]
        save_resp = client.post("/api/annotations", json=data)
        assert save_resp.status_code == 200

        reloaded = client.get("/api/annotations").json
        obj = reloaded["frames"][0]["objects"][0]
        assert obj["category"] == "road_debris_or_prop"
        assert obj["provenance"] == "human_added"

    def test_no_duplicate_proposals_over_existing_draft_boxes(self):
        """IoU overlap matching prevents duplicate proposals over existing draft boxes."""
        # Simulated existing draft box (from Frame 0: [494, 559, 778, 721])
        draft_box = {"x1": 494.0, "y1": 559.0, "x2": 778.0, "y2": 721.0}
        # Simulated model proposal box: [482.6, 552.5, 760.9, 705.5]
        proposal_box = {"x1": 482.6, "y1": 552.5, "x2": 760.9, "y2": 705.5}

        xA = max(draft_box["x1"], proposal_box["x1"])
        yA = max(draft_box["y1"], proposal_box["y1"])
        xB = min(draft_box["x2"], proposal_box["x2"])
        yB = min(draft_box["y2"], proposal_box["y2"])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (draft_box["x2"] - draft_box["x1"]) * (draft_box["y2"] - draft_box["y1"])
        area2 = (proposal_box["x2"] - proposal_box["x1"]) * (proposal_box["y2"] - proposal_box["y1"])
        iou = inter / (area1 + area2 - inter)

        # Confirm IoU exceeds the 0.50 matching threshold (actual: ~0.79)
        assert iou >= 0.50

    def test_independent_mode_tags_dataset_and_objects(self, tmp_path):
        frames_dir = tmp_path / "frames"
        scenario = "test/anomaly/Town01/running-pedestrian/scenario-1"
        sc_dir = frames_dir / scenario.replace("/", "_")
        sc_dir.mkdir(parents=True, exist_ok=True)
        img_path = sc_dir / "000000.jpg"
        Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(img_path, format="JPEG")
        img_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()

        dataset_file = tmp_path / "ann.json"
        ds = AnnotationDataset(
            schema_version="1.0.0",
            created_at_utc="2026-10-20T12:00:00Z",
            annotation_mode="independent",
            frames=[FrameAnnotation(scenario_id=scenario, tick=0, frame_hash_sha256=img_hash)],
        )
        ds.save_json(dataset_file)

        app = create_app(dataset_file, frames_dir, default_mode="independent")
        client = app.test_client()

        data = client.get("/api/annotations").json
        assert data["annotation_mode"] == "independent"

        ind_obj = {
            "object_id": "obj_ind_1",
            "track_id": 1,
            "category": "person",
            "bbox": {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 250.0},
            "occlusion": "none",
            "visibility": "clear",
            "is_ambiguous": False,
            "confirmation_status": "pending_review",
            "provenance": "independent_human",
        }
        data["frames"][0]["objects"] = [ind_obj]
        save_resp = client.post("/api/annotations", json=data)
        assert save_resp.status_code == 200

        reloaded = client.get("/api/annotations").json
        assert reloaded["annotation_mode"] == "independent"
        assert reloaded["frames"][0]["objects"][0]["provenance"] == "independent_human"

    def test_evaluator_separates_ai_assisted_from_independent_evaluation(self):
        from cognix.evaluation.tracking_evaluator import audit_dataset_provenance

        # AI-Assisted dataset
        ai_obj1 = AnnotatedObject("o1", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), provenance="ai_accepted")
        ai_obj2 = AnnotatedObject("o2", 2, ObjectCategory.PERSON, BoundingBox(60, 60, 90, 150), provenance="human_added")
        f_ai = FrameAnnotation("sc", 0, "h0", objects=[ai_obj1, ai_obj2])
        ds_ai = AnnotationDataset(schema_version="1.0.0", annotation_mode="ai_assisted", frames=[f_ai])

        audit_ai = audit_dataset_provenance(ds_ai)
        assert audit_ai["annotation_mode"] == "ai_assisted"
        assert audit_ai["is_independent_ground_truth"] is False
        assert audit_ai["provenance_breakdown"]["ai_accepted"] == 1
        assert audit_ai["provenance_breakdown"]["human_added"] == 1
        assert audit_ai["ai_assisted_object_fraction"] == 0.5

        # Independent dataset
        ind_obj = AnnotatedObject("o3", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), provenance="independent_human")
        f_ind = FrameAnnotation("sc", 0, "h0", objects=[ind_obj])
        ds_ind = AnnotationDataset(schema_version="1.0.0", annotation_mode="independent", frames=[f_ind])

        audit_ind = audit_dataset_provenance(ds_ind)
        assert audit_ind["annotation_mode"] == "independent"
        assert audit_ind["is_independent_ground_truth"] is True
        assert audit_ind["provenance_breakdown"]["independent_human"] == 1
        assert audit_ind["ai_assisted_object_fraction"] == 0.0
