"""Configured sources -> ordinary DecisionEngine -> generic dashboard payload."""
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
import sys
import threading
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from cognix.runtime.replay import SourceError
from dashboard.payload import build_payload
from dashboard.sources import synthetic_runtime, dataset_runtime

engine, dataset, agents, bridge, controller = synthetic_runtime(ROOT)
MODE = "SYNTHETIC"
SOURCE_ERROR = None
BUSY = False
LOCK = threading.RLock()
CURRENT_SCENARIO_NAME = "NORMAL"
TICK = [0]
LATENCY_HISTORY = []
history = []
MAX_HISTORY = 60

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(live_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

app = FastAPI(title="COGNIX Research Replay", version="0.3.0", lifespan=lifespan)

class ConnectionManager:
    def __init__(self):
        self.active = []
    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active:
            self.active.remove(ws)
    async def broadcast(self, payload):
        for ws in list(self.active):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(ws)

manager = ConnectionManager()

def clear_history():
    history.clear()
    LATENCY_HISTORY.clear()
    TICK[0] = 0

def available():
    if controller is None:
        raise SourceError(SOURCE_ERROR or "Source unavailable")
    return controller

def idle():
    if BUSY:
        raise HTTPException(409, "Source is processing a sample; pause and wait before reconfiguring")

def run_cognix_cycle(delta=1):
    global BUSY, SOURCE_ERROR, CURRENT_SCENARIO_NAME
    with LOCK:
        idle()
        selected, runtime = available(), bridge
        BUSY = True
    try:
        observation = selected.step(delta)
        cycle = runtime.run(observation)
        payload = build_payload(cycle, LATENCY_HISTORY)
        with LOCK:
            TICK[0], CURRENT_SCENARIO_NAME = int(observation.sample_id), observation.sequence_id
            LATENCY_HISTORY.append(cycle.runtime_ms)
            del LATENCY_HISTORY[:-100]
            history.append(payload)
            del history[:-MAX_HISTORY]
            SOURCE_ERROR = None
        return payload
    except Exception as exc:
        with LOCK:
            SOURCE_ERROR, selected.error, selected.playing = str(exc), str(exc), False
        raise
    finally:
        with LOCK:
            BUSY = False

class ScenarioRequest(BaseModel):
    scenario: str

class ModeRequest(BaseModel):
    mode: str
    split: str = "TEST"
    archive: str | None = None
    graph_strategy: str = "NoGraph"
    seed: int = 101
    max_samples: int = 300
    require_calibration: bool = False

class StepRequest(BaseModel):
    delta: int = 1

class RateRequest(BaseModel):
    rate: float

@app.get("/api/source/status")
def source_status():
    with LOCK:
        return {"mode": MODE, "busy": BUSY, "error": SOURCE_ERROR,
                "source": controller.status() if controller else None,
                "runtime": bridge.provenance if bridge else None,
                "calibration_available": bool(bridge and bridge.engine.pipeline.calibrator is not None),
                "connections": len(manager.active)}

@app.post("/api/source/mode")
def set_mode(req: ModeRequest):
    global engine, dataset, agents, bridge, controller, MODE, SOURCE_ERROR, CURRENT_SCENARIO_NAME
    with LOCK:
        idle()
        mode = req.mode.upper()
        if mode not in ("SYNTHETIC", "DATASET"):
            raise HTTPException(400, "SIMULATION and LIVE are future modes; choose SYNTHETIC or DATASET")
        MODE, controller, bridge, SOURCE_ERROR = mode, None, None, None
        CURRENT_SCENARIO_NAME = None
        clear_history()
        try:
            if mode == "SYNTHETIC":
                engine, dataset, agents, bridge, controller = synthetic_runtime(ROOT)
                CURRENT_SCENARIO_NAME = "NORMAL"
            else:
                engine, dataset, agents, bridge, controller = dataset_runtime(ROOT, req)
        except Exception as exc:
            SOURCE_ERROR = str(exc)
            raise HTTPException(422, SOURCE_ERROR) from exc
    return source_status()

@app.get("/api/dataset/scenarios")
def scenarios():
    with LOCK:
        try:
            return {"mode": MODE, "scenarios": available().source.sequences()}
        except SourceError as exc:
            raise HTTPException(503, str(exc)) from exc

@app.post("/api/dataset/select")
@app.post("/api/set_scenario")
def set_scenario(req: ScenarioRequest):
    global CURRENT_SCENARIO_NAME, SOURCE_ERROR
    with LOCK:
        idle()
        try:
            available().select(req.scenario)
        except SourceError as exc:
            raise HTTPException(422, str(exc)) from exc
        CURRENT_SCENARIO_NAME, SOURCE_ERROR = req.scenario, None
        clear_history()
    return {"status": "ok", "scenario": req.scenario}

@app.post("/api/dataset/play")
def play():
    with LOCK:
        try:
            selected = available()
            if selected.sequence_id is None:
                raise SourceError("Select a scenario before playback")
            if selected.error or SOURCE_ERROR:
                raise SourceError("Source failed; restart or reconfigure before playback")
            selected.playing = True
        except SourceError as exc:
            raise HTTPException(422, str(exc)) from exc
    return source_status()

@app.post("/api/dataset/pause")
def pause():
    with LOCK:
        if controller:
            controller.playing = False
    return source_status()

@app.post("/api/dataset/restart")
def restart():
    global SOURCE_ERROR
    with LOCK:
        idle()
        try:
            available().restart()
        except SourceError as exc:
            raise HTTPException(503, str(exc)) from exc
        SOURCE_ERROR = None
        clear_history()
    return source_status()

@app.post("/api/dataset/rate")
def rate(req: RateRequest):
    with LOCK:
        try:
            available().set_rate(req.rate)
        except SourceError as exc:
            raise HTTPException(422, str(exc)) from exc
    return source_status()

@app.post("/api/dataset/step")
def step(req: StepRequest):
    pause()
    try:
        return run_cognix_cycle(req.delta)
    except SourceError as exc:
        raise HTTPException(422, str(exc)) from exc

@app.get("/api/status")
def api_status():
    return {**source_status(), "status": "error" if SOURCE_ERROR else "ok", "tick": TICK[0],
            "scenario": CURRENT_SCENARIO_NAME, "version": "0.3.0"}

@app.get("/api/latest")
def api_latest():
    with LOCK:
        if SOURCE_ERROR:
            raise HTTPException(503, SOURCE_ERROR)
        if history:
            return history[-1]
        if MODE == "DATASET":
            raise HTTPException(409, "No sample yet; select a scenario and use Step or Play")
    return run_cognix_cycle()

@app.get("/api/history")
def api_history():
    with LOCK:
        return history[-50:]

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        if history and not SOURCE_ERROR:
            await ws.send_json(history[-1])
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)

async def live_loop():
    while True:
        if controller and controller.playing and not BUSY:
            try:
                payload = await asyncio.to_thread(run_cognix_cycle)
                await manager.broadcast(payload)
            except Exception as exc:
                await manager.broadcast({"mode": MODE, "error": str(exc), "payload_type": "SOURCE_ERROR"})
        await asyncio.sleep(1.0 / (controller.rate if controller else 1.0))

STATIC_DIR = ROOT / "dashboard/static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
