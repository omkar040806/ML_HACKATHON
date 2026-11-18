"""PathSnap FastAPI backend - runs on Laptop 1 (owner: Person 4).

Endpoints
  GET  /health   model + data status
  GET  /trips    trip ids available for evaluation (no labels)
  POST /predict  {"trip_id": "..."} -> prediction + server-side metrics

Run (Laptop 1):
  PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment
  PATHSNAP_EVAL_DIR       folder with the evaluator trip sub-folders (takes priority)
  PATHSNAP_API_PARTITION  partition name under data/raw if PATHSNAP_EVAL_DIR is unset
                          (default from config: eval_test; use student_test for rehearsal)
  PATHSNAP_MODEL_PATH     model bundle (default models/final_model.joblib)

Every request is logged to the console with trip id, client address,
timings and metric values so the evaluator can see the API working.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import Evaluation, PredictRequest, PredictResponse, RouteMetrics, TripInfo, TripList
from pathsnap.config import load_config, partition_dir
from pathsnap.io import TripFormatError, list_trips, load_reference_route, load_trip_input, trip_path
from pathsnap.pipeline import PathSnapPipeline, arc_segments, route_polylines, snapped_points

log = logging.getLogger("pathsnap.api")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [pathsnap.api] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _eval_dir(cfg: dict) -> Path:
    if os.environ.get("PATHSNAP_EVAL_DIR"):
        return Path(os.environ["PATHSNAP_EVAL_DIR"]).resolve()
    part = os.environ.get("PATHSNAP_API_PARTITION", cfg["api"]["partition"])
    return partition_dir(cfg, part)


def create_app() -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = load_config()
        state["cfg"] = cfg
        state["pipeline"] = PathSnapPipeline(cfg)
        state["eval_dir"] = _eval_dir(cfg)
        ids = list_trips(state["eval_dir"])
        log.info("serving %d trips from %s", len(ids), state["eval_dir"])
        if not ids:
            log.warning("no trips found - set PATHSNAP_EVAL_DIR to the evaluator trip folder")
        yield
        state.clear()

    app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health")
    def health():
        p = state["pipeline"]
        return {"status": "ok", "model": p.bundle.model_name, "model_path": p.model_path,
                "hmm_params_source": p.hmm_params_source, "trip_dir": str(state["eval_dir"]),
                "n_trips": len(list_trips(state["eval_dir"]))}

    @app.get("/trips", response_model=TripList)
    def trips():
        out = []
        for tid in list_trips(state["eval_dir"]):
            try:
                tr = load_trip_input(state["eval_dir"] / tid).track
                out.append(TripInfo(trip_id=tid, n_observations=len(tr), duration_s=float(tr[-1, 2] - tr[0, 2])))
            except TripFormatError as exc:
                log.warning("skipping %s: %s", tid, exc)
        log.info("GET /trips -> %d trips", len(out))
        return TripList(partition=state["eval_dir"].name, trips=out)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request):
        t0 = time.perf_counter()
        client = request.client.host if request.client else "?"
        log.info("POST /predict trip=%s from %s", req.trip_id, client)
        if req.trip_id not in list_trips(state["eval_dir"]):
            log.warning("unknown trip id %s", req.trip_id)
            raise HTTPException(status_code=404, detail=f"trip {req.trip_id} not found")
        d = trip_path(state["eval_dir"], req.trip_id)
        pipe: PathSnapPipeline = state["pipeline"]
        try:
            trip = load_trip_input(d)                 # track + nodes + arcs only
            res, pt = pipe.predict(trip)
        except TripFormatError as exc:
            log.error("bad trip %s: %s", req.trip_id, exc)
            raise HTTPException(status_code=422, detail=str(exc))

        # ---- server-side evaluation (after prediction; reference never leaves the server)
        ev = Evaluation(available=False, note="no reference route on server for this trip")
        t_eval = time.perf_counter()
        ref = load_reference_route(d, n_arcs=len(trip.arcs))
        if ref is not None:
            m = pipe.evaluate(pt, res, ref)
            ev = Evaluation(available=True, hmm=RouteMetrics(**m["hmm"]),
                            classifier_only=RouteMetrics(**m["classifier_only"]),
                            note="computed on the server against its internal reference route")
        t_eval = time.perf_counter() - t_eval

        g = pt.graph
        cap = state["cfg"]["api"]["max_network_segments"]
        timings = {k: round(float(v), 4) for k, v in res.timings.items()}
        timings["evaluation_s"] = round(t_eval, 4)
        timings["request_total_s"] = round(time.perf_counter() - t0, 4)
        resp = PredictResponse(
            trip_id=res.trip_id, model_name=pipe.bundle.model_name, hmm_params_source=pipe.hmm_params_source,
            n_observations=res.n_obs, n_matched_observations=int((res.hmm_obs_arc >= 0).sum()),
            n_hmm_segments=res.n_segments, n_unfilled_gaps=res.n_unfilled_gaps,
            predicted_route_arc_ids=[int(a) for a in res.hmm_route],
            gps=[[float(r[1]), float(r[0]), float(r[2])] for r in trip.track],
            predicted_route=route_polylines(res.hmm_route, g),
            classifier_only_segments=arc_segments(res.clf_route, g, cap),
            network_segments=arc_segments(pt.cands["arc_id"].unique(), g, cap),
            snapped_points=snapped_points(pt, res.hmm_obs_arc),
            evaluation=ev, timings_s=timings)
        if ev.available:
            log.info("RESULT trip=%s n_obs=%d | HMM acc=%.4f cont=%.4f routeF1=%.4f | clf-only acc=%.4f cont=%.4f "
                     "| %.2fs", res.trip_id, res.n_obs, ev.hmm.pointwise_accuracy, ev.hmm.route_continuity,
                     ev.hmm.route_f1, ev.classifier_only.pointwise_accuracy, ev.classifier_only.route_continuity,
                     timings["request_total_s"])
        else:
            log.info("RESULT trip=%s n_obs=%d (no reference) | %.2fs", res.trip_id, res.n_obs,
                     timings["request_total_s"])
        return resp

    return app


app = create_app()
