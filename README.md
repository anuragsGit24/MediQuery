# Experiment 7 — Preparing an LLM Application for Production-Ready Deployment

## Healthcare Knowledge RAG API

This project converts the healthcare knowledge-base RAG prototype from the earlier experiment into a FastAPI service. It exposes the retriever and generator through HTTP, validates requests, handles provider failures safely, records latency, returns source metadata, and provides automated tests.

The application can use either of these generation backends:

1. **Gemini cloud API** — the model runs on Google's infrastructure and the application sends prompts over the internet.
2. **Ollama local LLM** — the model runs on the local computer and the application calls Ollama at `localhost`.

The backend is selected through one environment variable:

```env
LLM_PROVIDER=gemini
```

Change it to `ollama` to use a local model without changing the Python code.

> **Healthcare disclaimer:** This application provides educational information from the supplied documents. It is not medical advice, diagnosis, treatment, or a replacement for a qualified healthcare professional.

---

## 1. Aim

The aim of this experiment was to prepare an existing customer-support/healthcare RAG application for API-based and production-oriented use by adding:

- A FastAPI HTTP interface.
- Request validation.
- Controlled error handling.
- Basic operational logging.
- Response-time measurement.
- Source metadata in responses.
- Endpoint and integration testing.
- A switchable cloud/local LLM backend.
- A safe fallback when the answer is not present in the knowledge base.

The main design principle was to keep the RAG pipeline separate from the LLM provider. The retriever finds relevant document chunks and creates the prompt; a provider adapter generates the answer. This makes it possible to change Gemini to Ollama without rebuilding the document index or retrieval logic.

---

## 2. Learning outcomes demonstrated

This implementation demonstrates:

- The difference between a notebook prototype and an API-oriented application.
- How to expose an existing RAG workflow through FastAPI.
- How to configure a cloud-hosted API or local Ollama LLM.
- How to validate and reject malformed requests.
- How to return controlled errors without leaking stack traces or secrets.
- How to measure server and client response time.
- How to test the application through automated tests and a live HTTP test client.
- Why model loading and vector-index creation should happen once at startup.
- Why unsupported questions should produce an uncertainty response instead of a fabricated answer.

---

## 3. Project structure

```text
EXP_7_HEALTHCARE_RAG/
├── app.py                              # FastAPI application and endpoints
├── support_rag.py                      # Loading, chunking, retrieval and generation
├── client_test.py                      # Live HTTP integration test suite
├── tests/
│   └── test_api.py                     # Mocked API/unit tests
├── knowledge_base/
│   ├── wellness.txt                    # Wellness knowledge source
│   ├── Medication_Safety.pdf           # Medication-safety knowledge source
│   └── faq.docx                        # Healthcare/TPA FAQ source
├── chroma_db/                          # Generated local vector-store data
├── test_reports/
│   ├── experiment7_live_test_report.txt
│   └── experiment7_live_test_report.json
├── Experiment_7_PE_LLM.pdf             # Experiment/lab manual
├── requirements.txt                    # Python dependencies
├── pytest.ini                          # Ensures project imports work under pytest
├── .env                                # Local secrets/configuration; not committed
└── .gitignore                          # Ignores secrets, caches and generated data
```

The `.env` file and generated Chroma database must not be committed. The repository should contain a safe `.env.example` template when sharing configuration instructions.

---

## 4. Technology choices

| Component | Choice | Reason |
|---|---|---|
| API framework | FastAPI | Type validation, automatic OpenAPI documentation, clear endpoint structure |
| ASGI server | Uvicorn | Lightweight server for local development and deployment |
| RAG framework | LangChain community integrations | Reuses document loaders and vector-store integrations |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, practical CPU embedding model |
| Vector store | Chroma with cosine distance | Local persistence and similarity search |
| Chunk size | 500 characters | Keeps retrieved context focused |
| Chunk overlap | 80 characters | Preserves context across chunk boundaries |
| Retrieval count | Top 3 | Limits prompt size while providing supporting context |
| Distance threshold | `0.75` | Rejects weak retrieval matches |
| Cloud generator | Gemini through `google-genai` | Simple API integration and no local GPU requirement |
| Local generator | Ollama HTTP API | Keeps inference on the local computer |
| HTTP client | `httpx` | Calls the Ollama local endpoint |
| Configuration | `python-dotenv` and environment variables | Keeps secrets and deployment settings out of source code |
| Test framework | Pytest | Automated unit/API testing |
| Live client | `requests` | Tests the running HTTP service independently |

---

## 5. Procedure followed

### Step 1 — Reused the existing RAG pipeline

The previous document-processing and retrieval workflow was kept in `support_rag.py`. The implementation:

1. Loads the TXT, PDF and DOCX files.
2. Rejects missing knowledge-base files explicitly.
3. Extracts readable text.
4. Splits documents into 500-character chunks with 80-character overlap.
5. Creates normalized sentence-transformer embeddings.
6. Stores chunks in a Chroma cosine-distance collection.
7. Retrieves the top three matching chunks for each question.
8. Rejects weak matches above `MAX_DISTANCE`.
9. Builds a grounded prompt containing source labels and context.

This avoided rebuilding the core RAG system and followed the lab instruction to reuse the earlier experiment.

### Step 2 — Created the FastAPI application

`app.py` provides:

- `GET /health`
- `POST /ask`
- Automatic Swagger documentation at `/docs`
- OpenAPI schema at `/openapi.json`
- Startup initialization through the FastAPI lifespan
- Centralized request-validation handling
- Controlled backend and internal errors

The embedding model and vector store are initialized once during application startup instead of once per request.

### Step 3 — Added input validation

The `question` field:

- Is required.
- Must be a string.
- Must contain at least one character.
- Must not be blank after trimming whitespace.
- Must not exceed 1,000 characters.

Malformed JSON, missing fields, null values, numbers, empty strings, whitespace-only strings and oversized questions are rejected with HTTP `422`.

### Step 4 — Added controlled errors

Provider and retrieval failures are converted into safe responses:

| Failure | API response |
|---|---|
| Request validation failure | HTTP `422` |
| RAG service not initialized | HTTP `503` |
| Retrieval or generation provider failure | HTTP `502` |
| Unexpected internal failure | HTTP `500` |

The client receives a generic message. The server logs the operational event, but the API does not expose a traceback, API key, private URL, or provider-specific secret.

### Step 5 — Added latency measurement and logging

Every successful `/ask` response contains `latency_ms`. The service logs:

- Request path.
- Whether the response was grounded.
- Server latency.
- Number of returned source items.

Secrets and complete private prompts are not logged.

### Step 6 — Added the provider switch

The provider selection is read at startup:

```python
self.llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
```

The generation path is then selected as follows:

```text
gemini  -> Google Gemini API
ollama  -> http://localhost:11434/api/generate
local   -> accepted alias for ollama
```

An unsupported provider produces an explicit configuration error instead of silently selecting a different model.

### Step 7 — Added automated and live tests

Two forms of testing were used:

1. **Mocked Pytest tests** — fast and repeatable; no external model call is needed.
2. **Live client tests** — exercise the running server, retrieval, provider, response schema, latency and fallback behavior.

The live client also covers malformed JSON, injection-like text, OpenAPI availability and multiple knowledge-base questions.

---

## 6. Configuration

### Gemini cloud configuration

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_private_key
GEMINI_MODEL=gemini-3.1-flash-lite
```

The API key must be supplied through the local environment or a deployment secret manager. It must never be hard-coded in Python, committed to Git, or displayed in screenshots.

### Ollama local configuration

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=120
```

Install Ollama, start it, and download a small model:

```powershell
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

The server normally listens at `http://localhost:11434`. The application calls:

```text
POST http://localhost:11434/api/generate
```

with `stream=false`, zero temperature, the configured model name and the grounded RAG prompt.

### Retrieval configuration

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_DISTANCE=0.75
TOP_K=3
CHROMA_COLLECTION=healthcare_rag
LOG_LEVEL=INFO
```

The exact model names and provider settings can be changed through environment variables without editing application logic.

---

## 7. Running the application on Windows

From the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the configuration template and edit it:

```powershell
Copy-Item .env.example .env
```

Start the API:

```powershell
uvicorn app:app --reload
```

Open:

- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Health endpoint: <http://127.0.0.1:8000/health>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>

Example request:

```json
{
  "question": "What is polypharmacy?"
}
```

Example response shape:

```json
{
  "answer": "In its most simplistic definition, polypharmacy means an individual on multiple medications.",
  "sources": [
    {
      "file_name": "Medication_Safety.pdf",
      "page": 6,
      "excerpt": "..."
    }
  ],
  "best_distance": 0.12,
  "latency_ms": 1952.5,
  "grounded": true
}
```

---

## 8. API behavior

### `GET /health`

Reports that the API process is running:

```json
{
  "status": "ok",
  "service": "healthcare-rag-api"
}
```

The endpoint does not prove that the LLM provider, vector store, external network, or all knowledge-base files are healthy. It is a lightweight process/readiness indicator.

### `POST /ask`

The endpoint:

1. Validates the question.
2. Retrieves relevant chunks.
3. Checks the similarity threshold.
4. Returns the insufficient-information fallback if retrieval is weak.
5. Sends grounded context to the configured generator.
6. Returns the answer, sources, distance, latency and grounded flag.

The fallback text is:

```text
The answer is not available in the provided documents.
```

The system prompt also instructs the model not to diagnose, prescribe, recommend medication changes, or use outside knowledge.

---

## 9. Cloud-hosted LLM versus local LLM

### Operational comparison

| Factor | Gemini cloud API | Ollama local LLM |
|---|---|---|
| Where inference runs | Provider infrastructure | Local computer |
| Internet required | Yes | No for generation after model download |
| API key | Required | Not required for local Ollama |
| Hardware requirement | Mostly CPU/RAM for retrieval; no local inference GPU required | Depends on local RAM, CPU/GPU and model size |
| Startup | Application can start after configuration; provider is external | Ollama must be installed, running and supplied with the model |
| Model storage | Provider-managed | Local disk |
| Data path | Prompt leaves the computer and reaches provider | Prompt remains on the local machine |
| Scaling | Easier to scale by calling provider | Limited by local resources unless separately deployed |
| Cost | Free/trial quotas may be limited and can change | No per-request API fee, but uses local hardware and electricity |
| Latency | Network latency plus provider queue/inference time | No network round trip, but CPU inference can be slow |
| Availability | Depends on quota, account, model and network | Depends on local process, model files and hardware |
| Privacy | Requires careful review of provider terms and data handling | Better control for sensitive local data, though the host must still be secured |
| Maintenance | Provider manages model serving | User manages Ollama, model files and resource usage |

### What was better about each choice

**Gemini was better for this recorded run** because it provided a convenient hosted generator without requiring a local inference model or GPU. It was practical for demonstrating a cloud API integration and produced successful live answers for the supplied questions.

**Ollama is better when local control is more important**. It avoids sending the prompt to a third-party API, works without a cloud API key, and is useful when internet connectivity or provider quota is unreliable. The trade-off is that the model must be downloaded and the computer must have enough memory and processing capacity.

The RAG architecture was deliberately designed so neither choice requires changing document loading, embeddings, retrieval, fallback logic or the API contract.

### Important distinction

Hosting FastAPI on a cloud platform does **not** automatically host the LLM there. The API process and the model runtime are separate. A cloud-hosted FastAPI instance can:

- Call a cloud LLM API.
- Call an Ollama process on the same machine if Ollama is deployed there.
- Call a separately hosted private model endpoint.

The deployment topology must be configured explicitly.

---

## 10. Testing procedure

### Unit/API tests

Run:

```powershell
python -m pytest -q
```

Result recorded during this experiment:

```text
8 passed, 1 warning
```

The warning was an existing deprecation warning from `langchain-community`; it did not fail the tests.

The tests cover:

- Health endpoint.
- Valid question.
- Empty question.
- Question longer than 1,000 characters.
- Controlled backend failure.
- Unsupported-question fallback.
- Ollama provider routing.
- Invalid provider configuration.

### Live integration tests

Start the API first:

```powershell
uvicorn app:app --reload
```

In a second terminal:

```powershell
python client_test.py
```

The client test writes:

- `test_reports/experiment7_live_test_report.txt`
- `test_reports/experiment7_live_test_report.json`

The helper uses a default timeout of 180 seconds, while health/OpenAPI checks use an 8-second override. The timeout override was fixed so it no longer passes two `timeout` keyword arguments to `requests`.

---

## 11. Recorded results

### Automated test result

| Test suite | Result | Interpretation |
|---|---:|---|
| `pytest -q` | **8 passed, 1 warning** | Mocked API and provider behavior passed |
| `client_test.py` live suite | **25 passed, 0 failed** | Running API passed health, validation, RAG, fallback, input-safety and response-contract checks |

### Live test summary

The live report was generated on **2026-10-24 18:52:21 UTC** against `http://127.0.0.1:8000`.

| Category | Tests | Passed | Failed |
|---|---:|---:|---:|
| Health and OpenAPI | 2 | 2 | 0 |
| Input validation | 7 | 7 | 0 |
| Relevant knowledge-base questions | 9 | 9 | 0 |
| Out-of-domain fallback | 5 | 5 | 0 |
| Script/SQL-like input | 1 | 1 | 0 |
| Response contract | 1 | 1 | 0 |
| **Total** | **25** | **25** | **0** |

### Observation table

| Test case | Expected result | Actual result | Pass/Fail |
|---|---|---|---|
| Health endpoint | API reports running | HTTP 200; `status=ok`; client latency about 22.54 ms | PASS |
| Supported question | Grounded answer with source metadata | Nine relevant questions returned HTTP 200, `grounded=true`, answers and source files | PASS |
| Unsupported question | Insufficient-information fallback | Five out-of-domain questions returned the configured fallback with `grounded=false` | PASS |
| Empty question | Validation error | HTTP 422 | PASS |
| Whitespace-only question | Validation error | HTTP 422 | PASS |
| Question over 1,000 characters | Validation error | HTTP 422 | PASS |
| Missing/null/non-string question | Validation error | HTTP 422 | PASS |
| Malformed JSON | Validation error | HTTP 422 | PASS |
| Backend unavailable/failure | Controlled error without secret details | Unit test returned controlled HTTP 502; no traceback or secret was exposed | PASS |
| Response-time measurement | Latency is recorded | `latency_ms` returned in every successful `/ask` response | PASS |
| OpenAPI documentation | Schema is available | HTTP 200 from `/openapi.json` | PASS |
| Injection-like input | API does not crash | Structured HTTP 200 fallback response | PASS |

### Live relevant-question observations

| Question area | Source returned | Grounded | Approximate server latency |
|---|---|---:|---:|
| Hydration | `wellness.txt` | Yes | 2,265 ms |
| Under-nutrition signs | `wellness.txt` | Yes | 4,590 ms |
| Protein healthy-living tips | `wellness.txt` | Yes | 12,044 ms |
| Cashless file submission timing | `faq.docx` | Yes | 1,523 ms |
| Provider Module | `faq.docx` | Yes | 1,729 ms |
| Physical claim-document submission | `faq.docx` | Yes | 1,528 ms |
| NHCX | `faq.docx` | Yes | 2,222 ms |
| Medication error definition | `Medication_Safety.pdf` | Yes | 11,955 ms |
| Polypharmacy | `Medication_Safety.pdf` | Yes | 1,953 ms |

These timings include local retrieval and model-generation behavior in the recorded environment. They are observations, not performance guarantees.

---

## 12. Answers to discussion/viva questions

### 1. What is the purpose of FastAPI in this experiment?

FastAPI exposes the RAG pipeline as an HTTP service. Other programs can send a question to `POST /ask` and receive a structured JSON response. It also provides request validation, automatic OpenAPI documentation, clear error responses and a health endpoint.

### 2. What is the difference between a local LLM and a cloud-hosted LLM API?

A local LLM runs on the user's computer through software such as Ollama. Its model files and inference workload are local, so it can provide better data locality and does not require a per-request cloud API call. A cloud-hosted LLM runs on a provider's infrastructure and is accessed over the internet using an API. Cloud inference usually reduces local hardware requirements but depends on network availability, account credentials, quotas and provider policies.

### 3. Does hosting FastAPI in the cloud automatically host the LLM there?

No. FastAPI is the API application; the LLM is a separate service. A cloud-hosted FastAPI process may call Gemini, call a separately hosted model, or call a local Ollama process only if Ollama is deployed and reachable in the selected infrastructure.

### 4. Why should an API key not be hard-coded in Python source?

Source code can be committed, copied, logged, backed up or exposed in screenshots. A hard-coded key could be used by another person, consume quota, create unexpected charges or expose private data. Environment variables and secret managers allow keys to be rotated without changing source code.

### 5. What is input validation, and why is a maximum question length useful?

Input validation checks that the request has the expected type, required fields and acceptable content before processing. A maximum length limits memory and token use, reduces accidental or abusive oversized requests, improves latency predictability and prevents unnecessarily large prompts.

### 6. What is the purpose and limitation of a health endpoint?

`GET /health` gives a quick indication that the API process is reachable and responding. It is useful for development and basic monitoring. In this implementation it is not a full dependency check: a successful health response does not prove that Gemini, Ollama, the vector store or every model dependency is available.

### 7. Why should the model and vector index not be loaded for every request?

Loading embeddings and rebuilding the vector index are expensive operations. Repeating them for every request increases latency, consumes memory and can cause inconsistent or unreliable behavior under concurrent traffic. Initializing reusable components during startup makes requests faster and more predictable.

### 8. What should RAG do when the answer is absent from its knowledge base?

It should acknowledge insufficient information rather than inventing an answer. This application returns:

```text
The answer is not available in the provided documents.
```

It also sets `grounded` to `false` and uses a retrieval-distance threshold to reject weak matches.

### 9. Name two reasons a cloud inference request might fail.

Examples include:

- Invalid, expired or missing API key.
- Exhausted free quota or rate limit.
- Temporary provider outage.
- Incorrect model name or unavailable model.
- Network timeout or DNS failure.
- Provider-side request-size or safety rejection.

### 10. Is this lab application fully production-ready? Name two additional requirements for a real public service.

No. It is production-oriented for the lab but not a complete public-service deployment. A real service should additionally implement authentication and authorization, HTTPS, rate limiting, secure secret management, structured monitoring/alerting, load testing, model evaluation, audit controls, backup/recovery procedures and stronger dependency health checks. It should also undergo privacy, security and clinical-safety review before handling real patient information.

---

## 13. Decisions made and why they were better

### Separation of retrieval and generation

The retriever and provider adapter are separate. This reduced the cost and risk of switching models and made provider-specific failures easier to isolate.

### Environment-based provider switch

The `LLM_PROVIDER` variable avoids code edits when changing between Gemini and Ollama. This is better than maintaining two copies of the application because it keeps behavior, prompt construction and API responses consistent.

### Conservative fallback

The system uses both similarity distance and an explicit fallback instruction. This is safer for healthcare education because a confident but unsupported answer is worse than acknowledging that the documents do not contain the answer.

### Source metadata in responses

Each response includes the source filename, optional page number and excerpt. This makes answers easier to inspect and supports groundedness review.

### Controlled provider errors

The implementation logs the failure server-side but gives the client a generic message. This avoids leaking provider details while preserving operational visibility.

### Reusable startup resources

Embedding and vector-store initialization happens once during startup. This improves response time and avoids rebuilding expensive resources per request.

### Explicit live-test evidence

The live client records actual responses in TXT and JSON rather than claiming that a request succeeded based only on code inspection. Semantic correctness still requires manual review of answers and source excerpts.

---

## 14. Precautions and limitations

1. **Do not commit secrets.** Keep `.env`, API keys and deployment credentials outside version control.
2. **Rotate exposed keys.** If a key appears in a terminal, screenshot, chat or repository, revoke and replace it.
3. **Do not send confidential data.** Do not submit real patient identifiers, medical records, financial data or personal information to a third-party cloud model.
4. **Treat answers as educational.** Do not use this application for diagnosis, prescribing or emergency decisions.
5. **Cloud quotas change.** Free/trial inference can be rate-limited, discontinued or changed by the provider.
6. **Model names change.** Confirm that the configured cloud model is available before testing.
7. **Ollama consumes resources.** Large local models can exceed RAM, disk or CPU/GPU capacity; start with a small model.
8. **Local does not automatically mean secure.** Secure the host, local network and model endpoint.
9. **The health endpoint is limited.** It reports process availability and is not a complete dependency health check.
10. **Latency varies.** First-request model warm-up, embedding initialization, provider queues, network conditions and local hardware affect timing.
11. **Chroma data is generated.** The local `chroma_db` directory can be rebuilt from the knowledge base and should not be treated as source code.
12. **No clinical validation was performed.** Passing HTTP and automated tests does not establish medical accuracy or clinical safety.
13. **The application is not a full public deployment.** Authentication, HTTPS, rate limiting, monitoring, load testing and recovery planning are still required.

---

## 15. Production-readiness gap analysis

| Area | Current lab implementation | Additional real-service requirement |
|---|---|---|
| Authentication | None | User/service authentication and authorization |
| Transport security | Local HTTP for development | HTTPS with certificate management |
| Secrets | `.env` locally | Managed secret store and rotation |
| Rate limiting | Not implemented | Per-user/IP/provider limits |
| Observability | Basic logs and latency | Metrics, tracing, alerts and dashboards |
| Health checks | Process-level `/health` | Dependency/readiness/liveness checks |
| Retrieval | Local Chroma and threshold | Versioned index, evaluation and refresh strategy |
| LLM safety | Grounded prompt and fallback | Safety evaluation, policy enforcement and human review |
| Privacy | No real sensitive data permitted | Data minimization, retention and compliance controls |
| Availability | Single development process | Redundancy, autoscaling and recovery plan |
| Testing | Unit and live functional tests | Load, security, regression and adversarial testing |

---

## 16. References

- FastAPI: <https://fastapi.tiangolo.com/>
- Uvicorn: <https://www.uvicorn.org/>
- Ollama: <https://docs.ollama.com/>
- Gemini API: <https://ai.google.dev/>
- Hugging Face Inference Providers: <https://huggingface.co/docs/inference-providers>
- LangChain: <https://python.langchain.com/docs/>
- Sentence Transformers: <https://www.sbert.net/>
- Chroma: <https://docs.trychroma.com/>

---

## 17. Final result

The existing healthcare RAG application was successfully exposed through FastAPI and prepared for production-oriented experimentation. The recorded implementation:

- Accepts validated questions through `POST /ask`.
- Reports API availability through `GET /health`.
- Reuses the existing TXT, PDF and DOCX knowledge base.
- Returns grounded answers and source metadata.
- Returns a safe fallback for unsupported questions.
- Measures response latency.
- Handles backend failures without exposing secrets or tracebacks.
- Supports Gemini cloud generation and Ollama local generation through configuration.
- Passed **8/8 automated tests**.
- Passed **25/25 live integration checks** in the recorded report.

The resulting system is suitable for the experiment and controlled demonstration. It should not be treated as a finished public healthcare service until the security, privacy, monitoring, availability and clinical-safety requirements in the gap analysis have been addressed.
