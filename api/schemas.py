"""Request/response models for the PathSnap API (owner: Person 4).

The response deliberately contains NO reference-route data: only the GPS
track (already known to the client), the prediction, nearby road geometry
and server-computed metric values.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TripInfo(BaseModel):
    trip_id: str
    n_observations: int
    duration_s: float


class TripList(BaseModel):
    partition: str
    trips: list[TripInfo]


class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^[A-Za-z0-9_\-]+$", max_length=64)


class RouteMetrics(BaseModel):
    pointwise_accuracy: float | None = None
    direction_agnostic_accuracy: float | None = None
    route_continuity: float | None = None
    route_precision: float | None = None
    route_recall: float | None = None
    route_f1: float | None = None
    route_len_precision: float | None = None
    route_len_recall: float | None = None
    route_len_f1: float | None = None
    route_mismatch_fraction: float | None = None
    edit_similarity: float | None = None
    n_labeled_obs: int | None = None
    predicted_route_arcs: int | None = None
    # (the number of reference arcs is deliberately NOT exposed; extra keys are ignored)


class Evaluation(BaseModel):
    available: bool
    hmm: RouteMetrics | None = None
    classifier_only: RouteMetrics | None = None
    note: str = ""


class PredictResponse(BaseModel):
    trip_id: str
    model_name: str
    hmm_params_source: str
    n_observations: int
    n_matched_observations: int
    n_hmm_segments: int
    n_unfilled_gaps: int
    predicted_route_arc_ids: list[int]
    gps: list[list[float]]                    # [lat, lon, t]
    predicted_route: list[list[list[float]]]  # polylines of [lat, lon]
    classifier_only_segments: list[list[list[float]]]
    network_segments: list[list[list[float]]]
    snapped_points: list[list[float]]
    evaluation: Evaluation
    timings_s: dict[str, float]
