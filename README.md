# BeamNG.tech + LiFePO₄ Battery Digital Twin (Hybrid) — Starter Repo

This is a minimal scaffold to run a hybrid battery aging twin connected to BeamNG.tech.

## Structure
- `twin_service/`: FastAPI microservice implementing a simple LiFePO₄ twin
- `beamng_client/`: BeamNGpy loop that streams telemetry and applies derates
- `scenarios/`: Example scenario configs (placeholders)
- `logs/`: Where your run logs can go

## Quickstart
1) Python deps: `pip install -r requirements.txt`
2) Start the twin: `cd twin_service && uvicorn api:app --reload --port 8008`
3) Configure BeamNG path in `beamng_client/config.yaml`
4) Run the client: `python beamng_client/stream.py`

See the PDF guide for deeper explanations and how to add ML.
 
