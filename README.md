---
title: SafarBot Voice API
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 10000
short_description: FastAPI voice API for STT, NLU, route search, and TTS responses
---

# SafarBot AI Service

FastAPI service for SafarBot NLU, normalization, dialogue flow, and backend tool calls.

## Structure

```text
app/
  main.py
  config.py
  schemas.py
  nlu/
  core/
  stt/
  utils/
assets/
requirements.txt
Dockerfile
render.yaml
```

## Environment

Use `.env.example` as the template for your local `.env`.
Set `MODEL_LOAD_ON_STARTUP=0` on low-memory hosts so the service can boot and pass `/readyz` before lazily loading NLU models on demand.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /chat`
- `POST /voice/text` (compatibility endpoint)
