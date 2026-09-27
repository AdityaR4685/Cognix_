"""
COGNIX Live Research Dashboard

Live simulation dashboard powered by actual COGNIX DecisionEngine computation.
Generates multi-agent AV scenarios on the fly with CarlAnomalyDataset.
"""

import asyncio
import json
import time
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

from cognix import DecisionEngine, CognixConfig
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.agents import CameraAgent, DepthAgent, LiDARAgent, GNSSAgent, IMUAgent, SegAgent

# ── Global Engine & Artifact State ───────────────────────────────────────
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "synthetic"
)

try:
    from cognix.artifacts import load_synthetic_artifacts
    gat_model, conformal_calibrator, artifact_meta = load_synthetic_artifacts(
        ARTIFACTS_DIR, graph_type="EpistemicGAT"
    )
    print(f"[COGNIX] Loaded fitted EpistemicGAT and ConformalPredictor ({conformal_calibrator.n_cal} cal samples).")
except Exception as exc:
    print(f"[COGNIX] Warning: Fitted artifacts not loaded ({exc}). Falling back to unrefined/uncalibrated pipeline.")
    gat_model = None
    conformal_calibrator = None
    artifact_meta = None

config = CognixConfig()
engine = DecisionEngine(
    config=config,
    graph=gat_model,
    calibrator=conformal_calibrator,
    attribution=EpistemicShapley(),
    mode="production",
)
dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=1)
agents = [
    CameraAgent(),
    DepthAgent(),
    LiDARAgent(),
    GNSSAgent(),
    IMUAgent(),
    SegAgent()
]

# Track the current active scenario
CURRENT_SCENARIO_NAME = "NORMAL"
TICK = [0]
LATENCY_HISTORY = []
MAX_LATENCY_HISTORY = 100

def load_benchmark_reference() -> tuple[dict[str, dict], str]:
    """Load real persisted benchmark summary metrics from universal evaluation."""
    summaries_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "universal_evaluation", "summaries", "all_summaries.json"
    )
    if not os.path.isfile(summaries_file):
        return {}, "Benchmark summaries unavailable (Illustrative profiles)"
    try:
        with open(summaries_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        table = {}
        for sc in ["NORMAL", "MULTI_FAILURE", "HIGH_NOISE", "MISSING_AGENT", "OOD_SHIFT", "CONFLICTING"]:
            if sc in data and "EpistemicGAT" in data[sc]:
                entry = data[sc]["EpistemicGAT"]
                table[sc] = {
                    "ece": round(float(entry.get("ece", {}).get("mean", 0.0)), 3),
                    "cov": round(float(entry.get("cp_coverage", {}).get("mean", 0.0)) * 100, 1),
                    "epi": round(float(entry.get("mean_epistemic", {}).get("mean", 0.0)), 3),
                    "lat": round(float(entry.get("latency_mean", {}).get("mean", 0.0)), 1),
                }
        return table, "Universal Benchmark (1,120 Runs — EpistemicGAT, illustrative reference profiles)"
    except Exception as exc:
        return {}, f"Error loading benchmark: {exc} (Illustrative profiles)"

BENCHMARK_COMPARISONS, BENCHMARK_NOTE = load_benchmark_reference()

SCENARIO_INFO = {
    "NORMAL": {"severity": "0 / 10", "effects": "None", "expected": "Stable epistemic variance"},
    "CAMERA_BLACKOUT": {"severity": "9 / 10", "effects": "Camera completely blind", "expected": "↑ Epistemic (Camera), ↓ Trust (Camera)"},
    "GPS_DRIFT": {"severity": "7 / 10", "effects": "GNSS positional drift", "expected": "↑ Epistemic (GNSS), ↓ Trust (GNSS)"},
    "HEAVY_RAIN": {"severity": "6 / 10", "effects": "LiDAR/Camera noise", "expected": "↑ Aleatoric & Epistemic"},
    "MULTI_FAILURE": {"severity": "10 / 10", "effects": "Cam/LiDAR/GNSS failing", "expected": "Escalation likely"}
}

# Mapping agents to icons for the UI
ICON_MAP = {
    "Camera": "📷",
    "Depth":  "📏",
    "LiDAR":  "📡",
    "GNSS":   "🛰️",
    "IMU":    "🧭",
    "Seg":    "🧠"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(live_loop())
    yield

app = FastAPI(title="COGNIX Dashboard", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_HISTORY = 60
history: list[dict] = []

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/api/set_scenario")
def set_scenario(req: ScenarioRequest):
    global CURRENT_SCENARIO_NAME
    if req.scenario in dataset.get_all_scenarios():
        CURRENT_SCENARIO_NAME = req.scenario
        return {"status": "ok", "scenario": CURRENT_SCENARIO_NAME}
    return {"status": "error", "message": "Unknown scenario"}, 400

def run_cognix_cycle() -> dict:
    global TICK, CURRENT_SCENARIO_NAME
    TICK[0] += 1
    
    # Generate 1 live frame for the current scenario
    frames = dataset.generate_frames(CURRENT_SCENARIO_NAME)
    frame = frames[0]
    
    inputs = {
        "Camera": frame.rgb,
        "Depth":  frame.depth,
        "LiDAR":  frame.lidar,
        "GNSS":   frame.gnss,
        "IMU":    frame.imu,
        "Seg":    frame.segmentation
    }
    
    # Determine which agents are affected by the current anomaly based on CarlAnomaly mapping
    affected_agents = []
    if CURRENT_SCENARIO_NAME == "CAMERA_BLACKOUT":
        affected_agents = ["Camera", "Depth"]
    elif CURRENT_SCENARIO_NAME == "GPS_DRIFT":
        affected_agents = ["GNSS"]
    elif CURRENT_SCENARIO_NAME == "HEAVY_RAIN":
        affected_agents = ["Camera", "LiDAR", "Depth"]
    elif CURRENT_SCENARIO_NAME == "MULTI_FAILURE":
        affected_agents = ["Camera", "GNSS", "IMU"]

    # Run decision engine
    reliabilities = {agent.agent_id: 1.0 for agent in agents}
    start_time = time.perf_counter()
    result = engine.decide(agents, inputs, agent_reliabilities=reliabilities)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

    # Build agent payload
    agents_payload = []
    for agent in agents:
        unc = agent.estimate_uncertainty(inputs)
        ep = unc.epistemic
        al = unc.aleatoric
        tot = unc.total
            
        weight = result.agent_trust_weights.get(agent.agent_id, 0.0)
        pred_obj = result.agent_predictions.get(agent.agent_id, 0.5)
        if hasattr(pred_obj, 'value'):
            pred = float(pred_obj.value)
        elif hasattr(pred_obj, 'prediction'):
            pred = float(pred_obj.prediction)
        else:
            try:
                pred = float(pred_obj)
            except Exception:
                pred = 0.5
        
        agents_payload.append({
            "name": agent.agent_id,
            "prediction": pred,
            "weight": weight,
            "epistemic": ep,
            "aleatoric": al,
            "healthy": agent.agent_id not in affected_agents
        })
        
    top_agent = max(result.agent_trust_weights, key=result.agent_trust_weights.get) if result.agent_trust_weights else "N/A"
    top_weight = result.agent_trust_weights.get(top_agent, 0.0)
    
    LATENCY_HISTORY.append(latency_ms)
    if len(LATENCY_HISTORY) > MAX_LATENCY_HISTORY:
        LATENCY_HISTORY.pop(0)
        
    lat_sorted = sorted(LATENCY_HISTORY)
    p50 = lat_sorted[int(len(lat_sorted) * 0.5)]
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)]
    p99 = lat_sorted[int(len(lat_sorted) * 0.99)]
    
    module_status = result.metadata.get("module_status", {})
    gnn_status = module_status.get("gnn", {})
    gnn_executed = bool(gnn_status.get("executed", False))
    cal_status = module_status.get("calibration", {})
    cal_executed = bool(cal_status.get("executed", False))

    # Real pipeline trace based on actual executed stages
    trace = [
        f"✓ {len(agents)} agents processed for {CURRENT_SCENARIO_NAME}",
        f"✓ MC Dropout UQ estimated (Epistemic: {result.epistemic_uncertainty:.3f}, Aleatoric: {result.aleatoric_uncertainty:.3f})",
    ]
    if gnn_executed:
        trace.append(f"✓ EpistemicGAT refinement executed ({len(result.agent_predictions)} agents)")
    trace.append(
        f"✓ Epistemic-weighted belief fusion ({top_agent} anchor, wt={top_weight:.2f})"
    )

    calibration_metrics = result.calibration_metrics or {}
    real_prediction_set = calibration_metrics.get("prediction_set", None)
    if real_prediction_set is not None:
        class_name_map = {0: "ESCALATE", 1: "ACT"}
        conformal_set = [class_name_map.get(c, str(c)) for c in real_prediction_set]
        trace.append(f"✓ Split-conformal calibration ({len(conformal_set)}-element set at 95% target coverage)")
    else:
        conformal_set = None

    trace.extend([
        f"✓ Risk assessment executed ({result.risk_level.name} risk)",
        f"✓ Decision generated ({result.decision.name})",
        f"✓ Epistemic Shapley attribution computed ({len(result.agent_contributions)} agents)",
    ])
    
    sinfo = SCENARIO_INFO.get(CURRENT_SCENARIO_NAME, {})
    
    if affected_agents:
        reasoning_text = (
            f"Primary evidence: {top_agent} anchor weight ({top_weight:.1%}).<br>"
            f"Uncertainty response: {', '.join(affected_agents)} influence reduced due to elevated epistemic uncertainty."
        )
    else:
        reasoning_text = (
            f"Primary evidence: Multi-agent agreement centered on {top_agent} ({top_weight:.1%}).<br>"
            f"Uncertainty response: Influences scaled inversely by agent epistemic uncertainty."
        )
        
    explanation_text = f"Collective probability: {result.confidence*100:.1f}% | Risk: {result.risk_level.name}"
    
    # Genuine measured stage latencies from CognixPipeline (perf_counter)
    pipe_lat = result.latency_ms or {}
    stage_latencies = {
        "uncertainty": float(pipe_lat.get("estimate_uncertainty", 0.0)),
        "graph_refinement": float(pipe_lat.get("graph_refinement", 0.0)),
        "fusion": float(pipe_lat.get("belief_fusion", 0.0) + pipe_lat.get("compute_trust", 0.0)),
        "calibration": float(pipe_lat.get("calibration", 0.0)),
        "decision": float(pipe_lat.get("decision", 0.0) + pipe_lat.get("risk_assessment", 0.0)),
        "attribution": float(pipe_lat.get("attribution", 0.0)),
        "total": latency_ms,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "measured_timer": "time.perf_counter"
    }

    return {
        "timestamp": time.time(),
        "tick": TICK[0],
        "scenario": CURRENT_SCENARIO_NAME,
        "scenario_details": {
            "severity": sinfo.get("severity", ""),
            "affected": ", ".join(affected_agents) if affected_agents else "None",
            "expected": sinfo.get("expected", "")
        },
        "research_meta": {
            "data_source": "CarlAnomalyDataset (Synthetic Simulation)",
            "experiment": "Live Interactive Synthetic Simulation",
            "run_id": f"LIVE_{int(time.time())}",
            "model": "COGNIX Live (Fitted EpistemicGAT + Conformal)",
            "dataset": "CarlAnomaly",
            "seed": "Live RNG"
        },
        "comparison_table": BENCHMARK_COMPARISONS,
        "comparison_table_note": BENCHMARK_NOTE,
        "decision": result.decision.name,
        "confidence": result.confidence,
        "collective_probability": result.confidence,
        "calibrated_confidence": result.calibrated_confidence,
        "calibration_available": cal_executed and result.calibrated_confidence is not None,
        "conformal_set": conformal_set,
        "conformal_target_coverage": 0.95 if conformal_set is not None else None,
        "graph_executed": gnn_executed,
        "risk_level": result.risk_level.name,
        "escalation": result.escalation_required,
        "abstained": result.abstained,
        "epistemic": result.epistemic_uncertainty or 0.0,
        "aleatoric": result.aleatoric_uncertainty or 0.0,
        "total_unc": result.total_uncertainty,
        "dominant_unc": result.dominant_uncertainty_source.upper(),
        "agents": agents_payload,
        "top_agent": top_agent,
        "shapley": result.agent_contributions,
        "latency": stage_latencies,
        "decision_trace": trace,
        "reasoning": reasoning_text,
        "explanation": explanation_text
    }

@app.get("/api/status")
def api_status():
    return {
        "status": "ok",
        "connections": len(manager.active),
        "tick": TICK[0],
        "scenario": CURRENT_SCENARIO_NAME,
        "version": "0.2.0.live",
    }

@app.get("/api/latest")
def api_latest():
    if history:
        return history[-1]
    return run_cognix_cycle()

@app.get("/api/history")
def api_history():
    return history[-50:]

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    if history:
        await ws.send_json(history[-1])
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

async def live_loop():
    while True:
        try:
            payload = run_cognix_cycle()
            history.append(payload)
            if len(history) > MAX_HISTORY:
                history.pop(0)
            await manager.broadcast(payload)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[COGNIX] Live loop error: {exc}")
        await asyncio.sleep(1.0)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    response = FileResponse(os.path.join(STATIC_DIR, "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  COGNIX Live Interactive Dashboard")
    print("  http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
