# Two-laptop deployment (owner: Person 4)

```
Laptop 2 (UI)                                  Laptop 1 (model + API)
streamlit run ui/app.py   ──GET /trips──────►  uvicorn api.main:app
  evaluator picks a trip  ──POST /predict───►    load track/nodes/arcs → features → model → HMM
  map + metrics           ◄──JSON───────────     score vs internal .route (server-side only)
```

## Laptop 1: API

1. Clone the repo, `pip install -r requirements.txt`.
2. Check that `models/final_model.joblib` and `results/hmm/best_hmm_params.json` exist
   (they are committed by Person 2 / Person 3).
3. Unzip `evaluator_test.zip` into `data/raw/` so the trips are in
   `data/raw/evaluator_test/<trip_id>/...` (the API's default), **or** set
   `PATHSNAP_EVAL_DIR` to wherever the evaluators put them.
4. Find the IP address: `ipconfig` (Windows) / `ip addr` or `ifconfig` (Linux/macOS).
5. Start the API and **keep this terminal visible**. The evaluator must see the request logs:

```bash
# Linux / macOS
PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
# Windows PowerShell
$env:PATHSNAP_EVAL_DIR="C:\path\to\evaluator_test"; uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Every request prints lines such as

```
... INFO [pathsnap.api] POST /predict trip=00000042 from 192.168.1.23
... INFO [pathsnap.api] RESULT trip=00000042 n_obs=1500 | HMM acc=... cont=... routeF1=... | clf-only acc=... | 1.20s
```

6. Allow inbound TCP 8000 in the firewall (Windows will prompt on first run).
7. Sanity check from Laptop 1: open `http://localhost:8000/docs`.

> **Internet is needed on Laptop 2 for the map.** Folium maps load the Leaflet
> JavaScript library from a CDN and the background tiles from OpenStreetMap. Without
> internet the map area stays blank (the metrics still work). A phone hotspot
> that both laptops join gives you the LAN connection *and* internet.

## Laptop 2: UI

```bash
pip install -r requirements-ui.txt
streamlit run ui/app.py -- --api http://<LAPTOP1_IP>:8000
```

The browser opens at `http://localhost:8501`. The sidebar shows whether the API is
reachable. The UI reads no data files; everything comes from the API.

## Rehearsal before the evaluation

Serve the student-test trips so the full flow can be checked end to end:

```bash
PATHSNAP_API_PARTITION=student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Checklist:
- [ ] both laptops on the same Wi-Fi / hotspot; `curl http://<IP>:8000/health` works from Laptop 2
- [ ] all 10 trips listed, and each one returns a map (try every one)
- [ ] screenshots: UI with map + metrics, and the Laptop 1 terminal with request logs (report §7)
- [ ] on demo day switch to the evaluator folder; nothing else changes

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says API not reachable | wrong IP, firewall, or different network; try a phone hotspot |
| `/trips` empty | `PATHSNAP_EVAL_DIR` wrong: it must contain one sub-folder per trip |
| `Model bundle not found` | run `scripts/finalize_model.py` or `git pull` |
| `different feature set` | features changed after training; rerun `finalize_model.py` |
| request times out on a huge trip | raise the timeout in the UI sidebar |
| map area blank, metrics fine | Laptop 2 has no internet (Leaflet CDN / OSM tiles) |
