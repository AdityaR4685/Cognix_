import json
import hashlib
from pathlib import Path
import pytest

from cognix.evaluation.ground_truth_schema import (
    AnnotationDataset,
    FrameAnnotation,
    AnnotatedObject,
    BoundingBox,
    ObjectCategory,
    OcclusionStatus,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CANDIDATE_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "track_identity_reconciliation_001"
    / "proposed_reconciled_pilot_annotations.json"
)

WORKING_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "m9_annotation_readiness_001"
    / "working_human_review_pilot_annotations.json"
)

LEDGER_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "track_identity_reconciliation_001"
    / "duplicate_retirement_ledger.json"
)

ADJUDICATION_LEDGER_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "track_identity_reconciliation_001"
    / "visual_adjudication_ledger.json"
)

EGO_EXCLUSION_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "ego_hood_correction_001"
    / "ego_hood_exclusion_ledger.json"
)

M8_CANONICAL_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "m8_ground_truth_tracking_001"
    / "pilot_annotations.json"
)

M9_IMMUTABLE_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "m9_annotation_readiness_001"
    / "baseline_pilot_annotations_immutable.json"
)

NORMAL_CONTROL_PATH = (
    REPO_ROOT
    / "reports"
    / "carla_runtime_v1"
    / "evaluation_runs"
    / "m9_annotation_readiness_001"
    / "normal_control_annotations.json"
)

EXPECTED_WORKING_HASH = "d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06"
EXPECTED_M8_HASH = "bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15"
EXPECTED_NORMAL_HASH = "3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestTrackIdentityCandidateValidation:
    """Validates the proposed reconciled candidate annotations and duplicate retirement ledger."""

    def test_original_working_file_and_baselines_are_strictly_unmodified(self):
        """Verifies that the working annotation file and all baseline queues remain byte-for-byte untouched."""
        assert compute_sha256(WORKING_PATH) == EXPECTED_WORKING_HASH
        assert compute_sha256(M8_CANONICAL_PATH) == EXPECTED_M8_HASH
        assert compute_sha256(M9_IMMUTABLE_PATH) == EXPECTED_M8_HASH
        assert compute_sha256(NORMAL_CONTROL_PATH) == EXPECTED_NORMAL_HASH

    def test_candidate_dataset_schema_and_integrity(self):
        """Verifies that the candidate loads cleanly into AnnotationDataset schema without validation errors."""
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        dataset = AnnotationDataset.from_dict(raw_data)
        assert len(dataset.frames) == 40

        total_objects = sum(len(fr.objects) for fr in dataset.frames)
        assert total_objects == 76, f"Expected 76 retained objects (81 minus 5 retired duplicates), got {total_objects}"

    def test_no_duplicate_track_ids_within_any_single_frame(self):
        """Ensures that no single frame has two objects with the same track_id."""
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for idx, fr in enumerate(raw_data["frames"]):
            track_ids = [o["track_id"] for o in fr["objects"]]
            assert len(track_ids) == len(set(track_ids)), (
                f"Duplicate track_id found in frame {idx} (tick {fr['tick']}): {track_ids}"
            )

    def test_duplicate_ai_proposals_retired_and_logged(self):
        """Verifies that exactly the 5 duplicate AI proposals are absent from candidate and present in ledger."""
        expected_retired_ids = {
            "obj_1792494631470_7_1",
            "obj_1792494651746_8_1",
            "obj_1792494668597_10_1",
            "obj_1792494683431_13",
            "obj_1792494819196_19", # Case B retired duplicate T19
        }

        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        candidate_object_ids = {
            o["object_id"] for fr in cand_data["frames"] for o in fr["objects"]
        }

        for ret_id in expected_retired_ids:
            assert ret_id not in candidate_object_ids, f"Retired object {ret_id} still present in candidate!"

        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            ledger_data = json.load(f)

        assert ledger_data["retired_duplicates_count"] == 5
        ledger_object_ids = {
            rec["retired_object"]["object_id"] for rec in ledger_data["retired_records"]
        }
        assert ledger_object_ids == expected_retired_ids

    def test_case_a_adjudication_integrity(self):
        """Verifies Case A object at tick 35 is person on provisional track 35 with partial occlusion."""
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        fr27 = cand_data["frames"][27]
        assert fr27["tick"] == 35
        case_a_objs = [o for o in fr27["objects"] if o["object_id"] == "obj_1792495448698"]
        assert len(case_a_objs) == 1
        obj = case_a_objs[0]
        assert obj["category"] == "person"
        assert obj["track_id"] == 35
        assert obj["occlusion"] == "partial"
        assert not obj["is_ambiguous"]

    def test_case_b_adjudication_and_track_continuity(self):
        """Verifies Case B retained T20 at tick 40 and carried forward track 20 to tick 45."""
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        # Tick 40 (Frame 08)
        fr08 = cand_data["frames"][8]
        assert fr08["tick"] == 40
        t20_objs = [o for o in fr08["objects"] if o["object_id"] == "obj_1792494830273_20"]
        assert len(t20_objs) == 1
        assert t20_objs[0]["track_id"] == 20
        # Verify T19 absent
        assert not any(o["object_id"] == "obj_1792494819196_19" for o in fr08["objects"])

        # Tick 45 (Frame 09)
        fr09 = cand_data["frames"][9]
        assert fr09["tick"] == 45
        t25_objs = [o for o in fr09["objects"] if o["object_id"] == "obj_1792494915966"]
        assert len(t25_objs) == 1
        assert t25_objs[0]["track_id"] == 20, f"Central pedestrian at tick 45 was not assigned track 20: {t25_objs[0]['track_id']}"

        # Foreground pedestrians remain separate tracks
        fg_tracks = {o["track_id"] for o in fr09["objects"] if o["object_id"] != "obj_1792494915966"}
        assert 12 in fg_tracks
        assert 16 in fg_tracks
        assert 20 not in fg_tracks

    def test_all_tracks_have_single_class_per_scenario(self):
        """Verifies that no track ID contains multiple different object classes within any scenario."""
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        scenario_tracks = {}
        for fr in cand_data["frames"]:
            sc = fr["scenario_id"]
            for o in fr["objects"]:
                scenario_tracks.setdefault(sc, {}).setdefault(o["track_id"], set()).add(o["category"])

        for sc, tracks in scenario_tracks.items():
            for tid, cats in tracks.items():
                assert len(cats) == 1, f"Scenario {sc} Track {tid} contains multiple classes: {cats}"

    def test_ego_hood_exclusions_remain_absent(self):
        """Ensures none of the 40 excluded CARLA ego-hood boxes are present in the candidate."""
        with open(EGO_EXCLUSION_PATH, "r", encoding="utf-8") as f:
            excl_data = json.load(f)

        excluded_ids = {rec["original_object_id"] for rec in excl_data["excluded_boxes"]}
        assert len(excluded_ids) == 40

        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        candidate_ids = {o["object_id"] for fr in cand_data["frames"] for o in fr["objects"]}
        for eid in excluded_ids:
            assert eid not in candidate_ids, f"Excluded ego hood box {eid} found in candidate!"

    def test_all_frames_remain_unconfirmed_pending_human_review(self):
        """Ensures that all 40 frames remain in unreviewed state so human reviewer must explicitly confirm."""
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        for idx, fr in enumerate(cand_data["frames"]):
            assert fr["review_state"] == "unreviewed", f"Frame {idx} review_state is not 'unreviewed'"
            assert fr["reviewer_id"] is None
            assert fr["confirmed_at_utc"] is None
