"""Dataset exploration (owner: Person 1).

Uses ONLY the train and student_test partitions.  Writes to results/eda/:

  inventory.csv            per-trip file sizes, counts, sampling, extent, network & label stats
  partition_summary.csv    aggregate statistics per partition
  data_quality.md          automatically detected anomalies + assumptions
  radius_miss_curve.csv    candidate miss rate as a function of search radius / K
  *.png                    distribution plots
  maps/eda_<trip>.html     Folium maps: GPS points, nearby road network, reference route

Usage:  python scripts/run_eda.py [--maps 00000001 00000013] [--n-jobs 2]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import load_config, partition_dir, results_dir
from pathsnap.dataset import build_partition_tables
from pathsnap.geo import LocalProjection, haversine_m
from pathsnap.graph import RoadGraph
from pathsnap.io import list_trips, load_reference_route, load_trip_input
from pathsnap.labels import route_continuity_breaks

log = logging.getLogger("eda")


def raw_trip_stats(partition: str, trip_dir: Path) -> dict:
    tid = trip_dir.name
    sizes = {f"size_{ext}_kb": round((trip_dir / f"{tid}.{ext}").stat().st_size / 1024, 1)
             for ext in ("track", "nodes", "arcs", "route") if (trip_dir / f"{tid}.{ext}").exists()}
    raw = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)
    trip = load_trip_input(trip_dir)
    tr = trip.track
    dt = np.diff(tr[:, 2])
    step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    proj = LocalProjection.from_points(tr[:, 0], tr[:, 1])
    g = RoadGraph(trip.nodes, trip.arcs, proj)
    x, y = proj.forward(tr[:, 0], tr[:, 1])
    route = load_reference_route(trip_dir, len(trip.arcs))
    comp = g.component_stats()
    ext_km = float(np.hypot(np.ptp(x), np.ptp(y)) / 1000)
    # worst-case equirectangular scale error at the extreme latitudes of this trip
    lat = tr[:, 1]
    scale_err = float(np.max(np.abs(np.cos(np.radians(lat)) / np.cos(np.radians(proj.lat0)) - 1)))
    row = {
        "partition": partition, "trip_id": tid, "n_files": len(list(trip_dir.iterdir())), **sizes,
        "n_obs_raw": len(raw), "n_obs": len(tr), "n_nonfinite": int((~np.isfinite(raw)).any(axis=1).sum()),
        "cleaning_issues": "; ".join(trip.issues),
        "t_start": tr[0, 2], "duration_s": tr[-1, 2] - tr[0, 2],
        "dt_median": float(np.median(dt)), "dt_max": float(dt.max()), "frac_dt_eq_1": float(np.mean(dt == 1)),
        "n_gaps_gt5s": int((dt > 5).sum()), "n_gaps_gt60s": int((dt > 60).sum()),
        "path_len_km": float(step.sum() / 1000),
        "max_step_m": float(step.max()), "max_speed_mps": float(np.max(step / np.maximum(dt, 1e-9))),
        "lon_c": proj.lon0, "lat_c": proj.lat0, "extent_km": ext_km, "proj_scale_err_max": scale_err,
        **comp,
        "route_len": 0 if route is None else len(route),
        "route_unique": 0 if route is None else len(np.unique(route)),
        "route_breaks": 0 if route is None else route_continuity_breaks(route, g),
        "route_len_km": 0.0 if route is None else float(g.arc_len[route].sum() / 1000),
    }
    return row


def folium_eda_map(trip_dir: Path, out: Path, max_arcs: int = 6000) -> None:
    import folium
    trip = load_trip_input(trip_dir)
    tr = trip.track
    proj = LocalProjection.from_points(tr[:, 0], tr[:, 1])
    g = RoadGraph(trip.nodes, trip.arcs, proj)
    route = load_reference_route(trip_dir, len(trip.arcs))
    x, y = proj.forward(tr[:, 0], tr[:, 1])
    from pathsnap.candidates import arcs_near_bbox
    near = arcs_near_bbox(g, x, y, 100.0)[:max_arcs]
    m = folium.Map(location=[proj.lat0, proj.lon0], zoom_start=14, tiles="OpenStreetMap")
    fg_net = folium.FeatureGroup(name=f"Road network near track ({len(near)} arcs)", show=True)
    for a in near:
        folium.PolyLine([(g.node_lat[g.arc_from[a]], g.node_lon[g.arc_from[a]]),
                         (g.node_lat[g.arc_to[a]], g.node_lon[g.arc_to[a]])],
                        color="#888888", weight=1, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)
    if route is not None:
        fg_r = folium.FeatureGroup(name="Reference route (.route)")
        pts = [(g.node_lat[g.arc_from[route[0]]], g.node_lon[g.arc_from[route[0]]])]
        pts += [(g.node_lat[g.arc_to[a]], g.node_lon[g.arc_to[a]]) for a in route]
        folium.PolyLine(pts, color="green", weight=5, opacity=0.8).add_to(fg_r)
        fg_r.add_to(m)
    fg_g = folium.FeatureGroup(name="GPS observations (.track)")
    step = max(1, len(tr) // 3000)
    for lon, lat, t in tr[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True, fill_opacity=0.7,
                            tooltip=f"t={t:.0f}s").add_to(fg_g)
    fg_g.add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="*", default=None, help="train trip ids to map (default: 3)")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    out = results_dir(cfg, "eda")

    rows = []
    for part in ("train", "student_test"):
        pdir = partition_dir(cfg, part)
        ids = list_trips(pdir)
        log.info("%s: %d trips in %s", part, len(ids), pdir)
        for tid in ids:
            rows.append(raw_trip_stats(part, pdir / tid))
    inv = pd.DataFrame(rows)

    # label / candidate statistics from the cached feature tables
    lab_rows = []
    curves = []
    for part in ("train", "student_test"):
        cands, obs = build_partition_tables(cfg, part, n_jobs=args.n_jobs)
        g = obs.groupby("trip_id")
        lab = pd.DataFrame({
            "labeled_frac": g["labeled"].mean(),
            "ref_dist_p50": g["ref_dist"].median(),
            "ref_dist_p95": g["ref_dist"].quantile(0.95),
            "no_candidate_frac": g["n_cands"].apply(lambda s: float((s == 0).mean())),
            "mean_cands": g["n_cands"].mean(),
            "cand_miss_rate": obs[obs.labeled].groupby("trip_id")["ref_in_cands"].apply(lambda s: 1 - s.mean()),
        }).reset_index()
        lab_rows.append(lab)
        lo = obs[obs.labeled]
        for r in (10, 15, 20, 30, 40, 50):
            curves.append({"partition": part, "radius_m": r,
                           "miss_rate_if_unbounded_K": float((lo.ref_dist > r).mean())})
        nearest = cands[(cands["rank"] == 0) & cands.obs_labeled]
        curves.append({"partition": part, "radius_m": "nearest_only_top1",
                       "miss_rate_if_unbounded_K": float(1 - nearest.label.sum() / len(lo))})
        # plots per partition
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].hist(np.clip(obs.ref_dist, 0, 100), bins=100)
        ax[0].set(title=f"{part}: distance GPS -> aligned reference arc", xlabel="m (clipped at 100)", ylabel="obs")
        ax[1].hist(obs.n_cands, bins=np.arange(-0.5, cfg["candidates"]["max_candidates"] + 1.5))
        ax[1].set(title=f"{part}: candidates per observation", xlabel="# candidates")
        ax[2].hist(cands.loc[cands.obs_labeled & (cands.label == 1), "rank"], bins=np.arange(-0.5, 8.5))
        ax[2].set(title=f"{part}: distance rank of reference arc", xlabel="rank (0 = nearest)")
        fig.tight_layout(); fig.savefig(out / f"labels_{part}.png", dpi=120); plt.close(fig)
    inv = inv.merge(pd.concat(lab_rows), on="trip_id", how="left")
    inv.to_csv(out / "inventory.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "radius_miss_curve.csv", index=False)

    num = inv.select_dtypes("number").columns
    summ = inv.groupby("partition")[list(num)].agg(["mean", "median", "min", "max"]).T
    summ.to_csv(out / "partition_summary.csv")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for part, c in (("train", "C0"), ("student_test", "C1")):
        s = inv[inv.partition == part]
        ax[0, 0].hist(s.n_obs, bins=30, alpha=.6, label=part, color=c)
        ax[0, 1].hist(np.log10(s.n_arcs), bins=30, alpha=.6, label=part, color=c)
        ax[0, 2].scatter(s.lon_c, s.lat_c, label=part, color=c, s=18)
        ax[1, 0].hist(s.frac_dt_eq_1, bins=30, alpha=.6, label=part, color=c)
        ax[1, 1].hist(np.log10(s.dt_max), bins=30, alpha=.6, label=part, color=c)
        ax[1, 2].hist(s.extent_km, bins=30, alpha=.6, label=part, color=c)
    ax[0, 0].set(title="GPS observations per trip"); ax[0, 1].set(title="log10(#arcs) per trip")
    ax[0, 2].set(title="Trip centroids (lon, lat)", xlabel="lon", ylabel="lat")
    ax[1, 0].set(title="Fraction of 1 s sampling steps"); ax[1, 1].set(title="log10(max time gap s)")
    ax[1, 2].set(title="Trajectory extent (km)")
    for a in ax.ravel():
        a.legend()
    fig.tight_layout(); fig.savefig(out / "inventory_overview.png", dpi=120); plt.close(fig)

    # sampling-interval distribution (train)
    dts = []
    for tid in inv[inv.partition == "train"].trip_id:
        tr = load_trip_input(partition_dir(cfg, "train") / tid).track
        dts.append(np.diff(tr[:, 2]))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(dts), bins=80, log=True)
    ax.set(title="train: sampling interval distribution", xlabel="log10(dt seconds)", ylabel="count (log)")
    fig.tight_layout(); fig.savefig(out / "sampling_intervals_train.png", dpi=120); plt.close(fig)

    # trajectory shapes (local metres) for 6 training trips
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for a, tid in zip(axs.ravel(), inv[inv.partition == "train"].trip_id.iloc[:6]):
        tr = load_trip_input(partition_dir(cfg, "train") / tid).track
        p = LocalProjection.from_points(tr[:, 0], tr[:, 1])
        x, y = p.forward(tr[:, 0], tr[:, 1])
        a.scatter(x, y, s=1, c=tr[:, 2], cmap="viridis")
        a.set_aspect("equal"); a.set(title=f"train {tid}", xlabel="x [m]", ylabel="y [m]")
    fig.tight_layout(); fig.savefig(out / "trajectories_train.png", dpi=120); plt.close(fig)

    # data-quality report
    q = ["# Data-quality record (generated by scripts/run_eda.py)", "",
         "Partitions analysed: train, student_test. eval_test was not opened.", ""]
    for part in ("train", "student_test"):
        s = inv[inv.partition == part]
        q += [f"## {part} ({len(s)} trips)", "",
              f"- GPS observations: total {int(s.n_obs.sum())}, median {int(s.n_obs.median())}, "
              f"min {int(s.n_obs.min())}, max {int(s.n_obs.max())}",
              f"- Non-finite GPS rows: {int(s.n_nonfinite.sum())}",
              f"- Trips with a gap > 60 s: {int((s.n_gaps_gt60s > 0).sum())}; largest gap {s.dt_max.max():.0f} s "
              f"(trip {s.loc[s.dt_max.idxmax(), 'trip_id']})",
              f"- Reference routes with physically disconnected consecutive arcs: "
              f"{int((s.route_breaks > 0).sum())} trips ({int(s.route_breaks.sum())} breaks)",
              f"- Duplicate arcs (same from/to): {int(s.n_duplicate_arcs.sum())}; self-loops: {int(s.n_self_loops.sum())}",
              f"- Trips with isolated nodes: {int((s.n_isolated_nodes > 0).sum())}; "
              f"median weak components {s.n_weak_components.median():.0f}",
              f"- Max trajectory extent {s.extent_km.max():.1f} km; worst equirectangular scale error "
              f"{100 * s.proj_scale_err_max.max():.2f} %",
              f"- Trips with < 90 % labelled observations: "
              f"{', '.join(s.loc[s.labeled_frac < 0.9, 'trip_id']) or 'none'}",
              f"- Median candidate miss rate {s.cand_miss_rate.median():.4f}, worst {s.cand_miss_rate.max():.4f} "
              f"(trip {s.loc[s.cand_miss_rate.idxmax(), 'trip_id']})", ""]
    q += ["## Assumptions and decisions", "",
          "- Columns are lon, lat, t (verified: lon in [-180,180], lat in [-90,90] for all trips).",
          "- Each trip is projected with a local equirectangular projection (see pathsnap/geo.py).",
          "- Arcs are straight segments between their nodes; duplicate arcs are merged to a canonical id.",
          "- Observation labels come from a monotone DP alignment of the track to the reference route "
          "(pathsnap/labels.py); observations farther than labels.max_label_distance_m are unlabelled.",
          "- The reference route is treated as reference information, not verified ground truth "
          "(routes with disconnected arcs above are evidence of its limitations).", ""]
    (out / "data_quality.md").write_text("\n".join(q))

    maps = args.maps if args.maps is not None else list(inv[inv.partition == "train"].trip_id.iloc[:3])
    for tid in maps:
        folium_eda_map(partition_dir(cfg, "train") / tid, out / "maps" / f"eda_{tid}.html")
        log.info("map written for %s", tid)
    log.info("EDA outputs in %s", out)


if __name__ == "__main__":
    main()
