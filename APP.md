# Local application

Generator Service Copilot is a four-view Streamlit application for the fictional service workflow. It runs locally and is not published or deployed.

## Views

- **Ask** classifies a service question, retrieves three approved chunks, generates a schema-validated response, shows exact citations, and forces escalation for low-confidence or invalidly cited output.
- **Inspect** scores an uploaded cable image with the selected ResNet patch-memory detector and displays its anomaly map. The result is explicitly a benchmark proxy and cannot declare equipment safe.
- **Prompt Lab** displays the fixed 96-generation development comparison and selected prompt.
- **Insights** displays fictional service-case cluster composition and representative cases without presenting them as company history.

The app loads the saved development-selected artifacts from checkpoints 2–4. Both final test splits remain sealed until checkpoint 6.

## Run locally

Start the pinned Redis service and application from the project root:

```bash
docker compose up -d redis
REDIS_URL=redis://127.0.0.1:6380/0 \
  .venv/bin/streamlit run src/generator_service_copilot/app.py
```

Without `REDIS_URL`, the Ask view uses local hybrid retrieval and reports caching as unavailable. If Redis becomes unavailable after startup, retrieval falls back locally and cache operations fail open. If generation fails or returns invalid JSON, the app returns an explicit low-confidence escalation instead of partial model text.

The local instruction model, sentence embedder, ResNet weights, and MVTec images must be present in their ignored cache/data locations. No OpenAI key is required. The optional OpenAI provider is contract-tested but is not exposed as the default application provider and has not been called live.

## Redis behavior

Response keys include the normalized question, provider, and document-corpus checksum. Values use a 15-minute TTL. A document-version change creates a new namespace, and the current version can be invalidated explicitly. The Redis container is intentionally ephemeral; a restart produces a safe cold-cache miss.

Live verification against Redis 8.2.3 confirmed cache miss/write/hit, one-second test expiry, current-version invalidation, document-version isolation, empty cache after restart, vector retrieval, a cached repeated question, and health reporting. Evidence is stored under `artifacts/app/`.

## Safety and privacy

The application uses fictional text data and a public non-commercial image benchmark. It does not diagnose real equipment, approve return to service, or replace qualified review. Uploaded images are written only to a temporary file for local inference. No analytics, public deployment, remote logging, or paid service is enabled.
