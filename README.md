# MP275 - Bakery Bread Sales (Surat)

**Group:** G121  **Problem:** MP275  **Course:** SCSE3040 MLOps  **Faculty:** Pulkit Dwivedi  **Batch:** 59

| Member | Enrolment no. |
|---|---|
| Rajkumar | S24CSEU1796 |
| Ritik Kumar | S24CSEU1795 |
| Aaditya Nagar | S24CSEU1840 |
| Achal Tyagi | S24CSEU1598 |

Client: **Surat Bakers' Guild**. Goal: predict `loaves_sold` per outlet per day (regression, metric = MAE).

## What is in this repository

| Path | Purpose |
|---|---|
| `src/mp275/data_checks.py` | the seven data checks (every number in `findings.yaml` comes from here) |
| `src/mp275/cleaning.py` | cleaning rules: drop leak column, fix `day_type` spellings, remove the stuck-sensor run |
| `src/mp275/modeling.py`, `train.py` | pipelines, v1 training with MLflow tracking and registry |
| `src/mp275/monitoring.py`, `retrain.py` | PSI, MAE per outlet, v2 retrain, promotion rule, holdout prediction |
| `src/mp275/make_findings.py` | writes `reports/findings.yaml` |
| `src/mp275/log.py` | JSON logging to stdout |
| `app/main.py` | FastAPI service (`POST /predict`, `GET /health`) with the client's two rules in the schema |
| `tests/` | pytest tests (including `TestClient` tests) - they need no CSV files |
| `Dockerfile`, `.dockerignore` | image `mp275-api:1.0` |
| `.github/workflows/ci.yml` | runs pytest on every push |
| `k8s/` | Deployment `mp275-api` (4 replicas) and Service `mp275-svc` (80 -> 8000) |
| `params.yaml` | paths, model settings, monitoring settings (read by the code with `yaml.safe_load`) |

## Data findings (details in `reports/findings.yaml`)

* `leaking_column`: **loaves_in_closing_register** (missing from holdout, correlation 0.9988 with the target)
* `stuck_sensor`: **shelf_hours** stuck at 9.5 from 2026-03-04 to 2026-03-22
* `spelling_variants`: **day_type** has 17 spellings but 3 real levels (weekday, weekend, festival)
* `unit_mixup`, `sentinel_value`, `duplicate_rows`, `target_typos`: not present

## How to reproduce

```bash
pip install -r requirements.txt
# put the original train.csv, later.csv, holdout_features.csv in data/raw/ (never open train.csv in Excel)
export PYTHONPATH=src
python -m mp275.data_checks      # the seven checks
python -m mp275.train            # v1: 3 MLflow runs + registered version 1
python -m mp275.retrain          # PSI, MAE per outlet, v2, decision, @champion, holdout predictions
python -m mp275.make_findings    # reports/findings.yaml
pytest                           # tests
```

## Results

* v1 = Ridge regression (lowest validation MAE 17.5 on cleaned train data).
* `later.csv` shows heavy drift: `orders_previous_day` PSI = 3.84 (all other columns below 0.04). College Gate is served worst (MAE 147).
* v2 = same model, trained on train + first 70% of `later.csv` (recent rows up-weighted). On the last 30%: v1 MAE 76.7, v2 MAE 18.7 (75.6% lower), so by the client's 5% rule **v2 is promoted** and is `@champion`.
* `predictions/holdout_predictions.csv` is produced by the champion.

## Run the service

```bash
uvicorn app.main:app --port 8000
docker build -t mp275-api:1.0 .
docker run -p 8000:8000 mp275-api:1.0
kubectl apply -f k8s/
```

Example request:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"price_rs":45,"temp_c":25,"footfall_index":60,"competitor_shops":3,"shelf_hours":9.5,"orders_previous_day":300,"day_type":"weekend","outlet":"Mall Road"}'
```
