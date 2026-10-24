"""HTTP replay smoke check. Dataset configuration is explicit and label-blind."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--archive")
    parser.add_argument("--split", choices=["TRAIN", "TEST"], default="TEST")
    parser.add_argument("--method", choices=["NoGraph", "StandardGAT", "EpistemicGAT"], default="NoGraph")
    parser.add_argument("--output")
    args = parser.parse_args()
    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(args.base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.loads(response.read())
    request("/api/dataset/pause", {})
    if args.archive:
        request("/api/source/mode", {"mode": "DATASET", "split": args.split, "archive": args.archive,
                                     "graph_strategy": args.method, "seed": 101, "max_samples": 2})
    listing = request("/api/dataset/scenarios")
    scenario = listing["scenarios"][0]
    request("/api/dataset/select", {"scenario": scenario})
    first = request("/api/dataset/step", {"delta": 1})
    second = request("/api/dataset/step", {"delta": 1})
    previous = request("/api/dataset/step", {"delta": -1})
    assert (first["tick"], second["tick"], previous["tick"]) == (1, 2, 1)
    assert request("/api/latest")["sample_id"] == "1"
    assert len(request("/api/history")) == 3
    assert not request("/api/dataset/rate", {"rate": 2})["source"]["playing"]
    assert request("/api/dataset/play", {})["source"]["playing"]
    request("/api/dataset/pause", {})
    request("/api/dataset/restart", {})
    # Leave a paused first frame available for inspection in the browser.
    final = request("/api/dataset/step", {"delta": 1})
    assert final["mode"] == ("DATASET" if args.archive else listing["mode"])
    assert final["ground_truth"] is None
    assert final["provenance"]["label_visible_to_inference"] is False
    if args.archive:
        assert final["active_agents"] == ["Camera", "IMU", "Seg"]
        assert final["calibrated_confidence"] is final["conformal_set"] is None
        assert not final["calibration_available"]
    receipt = {"status": "PASS_HTTP_REPLAY", "source_status": request("/api/source/status"),
               "controls": ["mode", "scenarios", "select", "next", "previous", "rate", "play", "pause", "restart"],
               "payload": final}
    print(json.dumps({"status": receipt["status"], "mode": final["mode"], "scenario": scenario,
                      "agents": final["active_agents"], "graph": final["graph_strategy"],
                      "decision": final["decision"], "calibration": final["calibration_available"]}))
    if args.output:
        Path(args.output).write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
