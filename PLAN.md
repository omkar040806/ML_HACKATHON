# PathSnap: commit plan

**5 commits in total**: one per person, in dependency order, plus one final team
commit. Each person commits **only after** their scripts have run on their own machine
and their tests pass, so every commit contains working code *and* the results it
produced. That keeps the history short, honest and easy for evaluators to verify.

```
 C1  Person 1  data layer + EDA + shared docs      (needs: nothing)
 C2  Person 2  models + comparison + Bayesian opt   (needs: C1)
 C3  Person 3  HMM + Viterbi + route evaluation     (needs: C2 model file)
 C4  Person 4  pipeline + API + UI + Folium maps    (needs: C3 HMM params)
 C5  Team      report PDF + screenshots + AI note   (needs: C4 + rehearsal)
```

While waiting for your turn, **read and understand your own bundle** (see
`docs/RESPONSIBILITIES.md`): run your tests (they use synthetic data and need only C1),
trace the code and prepare your report section. Waiting for the previous commit
before you commit is what keeps the count at five.

---

## Step 0: one-time setup (Person 1, ~5 min)

1. Create a **public** GitHub repo `pathsnap` (no README, no .gitignore. They come in C1).
2. Settings → Collaborators → add Persons 2, 3, 4.
3. Everyone (later):
   ```bash
   git clone https://github.com/<owner>/pathsnap.git && cd pathsnap
   python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt                         # after C1 exists
   ```
4. Everyone puts the dataset at `data/raw/train/…` and `data/raw/student_test/…`
   (git-ignored, **never commit data**).
5. Everyone sets their real name/email: `git config user.name "…"; git config user.email "…"`.

---

## C1: Person 1 (data layer, EDA, shared docs)

```bash
# copy your bundle contents into the repo root (keep the folder structure)
pytest -q tests/test_data_layer.py          # must pass
python scripts/build_features.py --n-jobs 4
python scripts/run_eda.py
git add .gitignore README.md PLAN.md requirements.txt pytest.ini configs docs \
        pathsnap/__init__.py pathsnap/config.py pathsnap/io.py pathsnap/geo.py pathsnap/graph.py \
        pathsnap/labels.py pathsnap/candidates.py pathsnap/features.py pathsnap/dataset.py \
        scripts/run_eda.py scripts/build_features.py tests/conftest.py tests/test_data_layer.py \
        results/eda
git status            # check: no data/ files, nothing from other bundles
git commit -m "Data layer: loading, projection, road graph, route-label alignment, candidates, features, EDA"
git push origin main
```
Then tell the team: **"C1 pushed."**

## C2: Person 2 (models and optimisation)

```bash
git pull
# copy your bundle in
pytest -q tests/test_models.py
python scripts/build_features.py --n-jobs 4     # rebuilds the git-ignored cache locally
python scripts/train_compare.py                 # longest step (KNN/RF); check run_log.txt
python scripts/bayes_opt.py                     # uses results/models/selection.json
python scripts/finalize_model.py                # writes models/final_model.joblib
python scripts/plot_model_results.py
git add pathsnap/models scripts/train_compare.py scripts/bayes_opt.py scripts/finalize_model.py \
        scripts/plot_model_results.py tests/test_models.py results/models results/bayes_opt models
git commit -m "Models: 8-model GroupKFold comparison, Optuna TPE optimisation, tuned final model"
git push
```
If `selection.json` picks a model other than the one you expect, that is fine. The
scripts follow the recorded evidence. Mention it in the report.

Cost notes: if the selected model is a Random Forest, each Bayesian trial takes
~0.5–3 min on 2 cores (30 trials ≈ 45 min); faster on more cores. All long scripts can
be restarted without losing work: `train_compare.py --resume` keeps finished models,
and `bayes_opt.py` continues the same SQLite study. `finalize_model.py` prints the model
file size; if it says > 90 MB, use Git LFS (`git lfs install; git lfs track "models/*.joblib"`;
commit `.gitattributes` too) because GitHub rejects files > 100 MB.

## C3: Person 3 (HMM and route reconstruction)

```bash
git pull                                        # needs models/final_model.joblib from C2
# copy your bundle in
pytest -q tests/test_hmm.py
python scripts/build_features.py --n-jobs 4
python scripts/tune_hmm.py                      # train trips, out-of-fold probabilities
python scripts/evaluate_hmm.py                  # student-test comparison
git add pathsnap/hmm scripts/tune_hmm.py scripts/evaluate_hmm.py tests/test_hmm.py results/hmm
git commit -m "HMM/Viterbi: classifier emissions, network transitions, gap handling, route evaluation"
git push
```

## C4: Person 4 (integration, API, UI, maps)

```bash
git pull                                        # needs C2 model + C3 HMM params
# copy your bundle in
pytest -q                                       # whole suite must pass now
python scripts/make_maps.py
# rehearsal (docs/DEPLOYMENT.md): API on Laptop 1 with PATHSNAP_API_PARTITION=student_test,
# UI on Laptop 2, click through all 10 trips, take screenshots into docs/screenshots/
git add pathsnap/pipeline.py pathsnap/viz api ui scripts/make_maps.py tests/test_api.py \
        requirements-ui.txt docs/DEPLOYMENT.md results/maps docs/screenshots
git commit -m "Integration: inference pipeline, FastAPI backend, Streamlit+Folium UI, report maps"
git push
```

## C5: Team (final, after the rehearsal works)

One person (agree who) commits:
```bash
git pull
git add docs/report/PathSnap_Technical_Report.pdf docs/AI_USE_DISCLOSURE.md docs/screenshots
git commit -m "Technical report, screenshots and AI-use disclosure"
git push
```

---

## Rules that keep the commit count low and the history clean

* **Do not commit work-in-progress.** Commit once, when your scripts ran and tests pass.
* **Never commit** `data/` (raw or processed), `.venv/`, secrets, or `YOUR_TASKS.md`.
* **If you find a bug in an earlier person's file**, fix it inside *your* commit,
  name it in the commit message (e.g. `…; fix: labels.py off-by-one (with P1)`), and tell
  the owner so they can explain it in the viva. Don't make a separate commit for it.
* **If you must change a shared interface** (`docs/INTERFACES.md`), agree with the
  affected owner first and update the doc in the same commit.
* **If a push is rejected** (someone pushed first): `git pull --rebase`, re-run tests, push.
* **Evaluator data**: copy it to Laptop 1 only on demo day; it is git-ignored. Never
  run training/tuning scripts on it (`build_partition_tables` refuses `eval_test`).
* **Results must match the code.** If you change code after running, re-run the affected
  script before committing. The report must only quote numbers from committed files.

## Timeline suggestion (hackathon day)

| Time | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| T+0 | repo + C1 | read bundle, test with synthetic data | read bundle | read bundle, install UI deps on Laptop 2 |
| T+0:30 | report §2 | C2 runs (≈30–60 min) | prepare §5 | network test between laptops |
| T+1:30 | help P2/P3 | report §3–4 | C3 (≈10–15 min) | |
| T+2:00 | | | report §5 | C4 + rehearsal |
| T+3:00 | everyone: review report, assemble PDF → C5 | | | |
