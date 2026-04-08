---
title: Oxidizer Rust Env
emoji: 🦀
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
  - rust
  - reinforcement-learning
  - swe-agent
  - ci-cd
---

# Oxidizer — Rust Build Fixer RL Environment

An [OpenEnv](https://huggingface.co/openenv)-compatible reinforcement learning environment where an AI agent fixes broken Rust projects by editing `Cargo.toml` and `src/main.rs` until `cargo check` succeeds.

## Motivation

Fixing Rust compilation errors is a real-world software engineering task that developers face daily. Unlike toy environments, this models a genuine developer workflow: read compiler diagnostics, identify root causes (missing dependencies, wrong feature flags, syntax errors), apply targeted fixes, and verify the build. The environment is designed following [AIDLC](https://github.com/awslabs/aidlc-workflows) principles with adaptive difficulty, quality gates, and structured audit logging.

## Tasks (5 levels, Easy to Expert)

| ID | Name | Difficulty | Error Type | Min Steps |
|----|------|-----------|------------|-----------|
| 0 | `missing_rand_dependency` | Easy | `rand` crate absent from `Cargo.toml` | 1 |
| 1 | `serde_feature_missing` | Medium | `serde` present but missing `features=["derive"]` | 1 |
| 2 | `syntax_and_dependency_error` | Hard | Missing semicolon + `reqwest` absent | 2 |
| 3 | `multiple_missing_dependencies` | Medium-Hard | Both `chrono` and `regex` absent | 1 |
| 4 | `cross_file_multi_error` | Expert | serde derive missing + `serde_json` absent + missing semicolon | 2+ |

### Difficulty Progression
- **Easy/Medium**: Single error type, single file edit fixes it
- **Hard/Medium-Hard**: Multiple errors but can be fixed in 1-2 targeted edits
- **Expert**: Errors span both files — agent must plan a multi-step repair strategy

## Action Space

```python
class Action(BaseModel):
    file_to_edit: Literal["Cargo.toml", "src/main.rs"]  # which file to overwrite
    new_content: str  # complete new file content (max 128 KB)
```

The agent edits **one file per step** by submitting the complete new content. This models the real-world pattern of editing a file and running `cargo check` to see if the fix worked.

## Observation Space

```python
class Observation(BaseModel):
    compiler_output: str   # stdout+stderr from cargo check (ANSI stripped)
    cargo_toml_content: str  # current Cargo.toml
    main_rs_content: str     # current src/main.rs
```

The agent sees the full compiler diagnostics including error codes (`E0432`, `E0277`, etc.) and the current file contents, enabling it to reason about what to change.

## Reward Function (Partial Progress)

The reward provides **gradient signal over the full trajectory** — not just binary pass/fail:

| Condition | Score | Done |
|-----------|-------|------|
| `cargo check` passes, no warnings | **1.00** | true |
| `cargo check` passes with N warnings | **max(0.95, 1.0 - N*0.01)** | true |
| Errors reduced from initial count | **min(0.9, reduction_ratio * 0.9)** | false |
| No error reduction or regression | **0.00** | false |

**Example trajectory on Task 4 (Expert):**
```
reset   → 5 errors (serde derive, serde_json missing, syntax)
step 1  → fix Cargo.toml (add derive + serde_json) → 1 error remaining → reward 0.72
step 2  → fix main.rs (add semicolon) → 0 errors → reward 1.00, done=true
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page with endpoint map |
| `/health` | GET | Liveness check |
| `/metadata` | GET | Environment name + description |
| `/schema` | GET | Action / Observation / State JSON schemas |
| `/tasks` | GET | List all 5 tasks with descriptions |
| `/reset` | POST | Reset to a task (`{"task_id": 0-4}` or empty body) |
| `/step` | POST | Apply one file edit |
| `/state` | GET | Current environment state |
| `/mcp` | POST | JSON-RPC 2.0 handshake |
| `/openapi.json` | GET | Full OpenAPI schema |

## Setup & Usage

### Local (Python)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn env:app --port 7860

# Test it:
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": 0}'
```

### Docker
```bash
docker build -t oxidizer .
docker run -p 7860:7860 oxidizer
```

### Inference (baseline agent)
```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python3 inference.py --task 0    # single task
python3 inference.py             # all 5 tasks
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | For inference | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | For inference | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | For inference | — | API authentication token |
| `API_KEY` | Optional | — | Alternative to HF_TOKEN |
| `LOCAL_IMAGE_NAME` | Optional | — | Docker image for local testing |
| `API_KEY` (server) | Optional | — | Server-side auth (X-Api-Key header) |
| `DEBUG` | Optional | `false` | Enable Swagger UI at `/docs` |

## Security (OWASP Top 10:2025)

- **A01**: Optional API_KEY auth middleware
- **A02**: Swagger UI disabled in production
- **A03**: `extra="forbid"` on all request models; `new_content` max 128KB
- **A04**: Credentials from env vars only, never logged
- **A05**: `file_to_edit` is a Pydantic `Literal` — no injection possible
- **A09**: Structured logging; opaque error IDs in HTTP 500s
- **A10**: Internal details never exposed to clients
- **ASI04**: Compiler output sanitised before LLM prompt embedding

## AIDLC Workflow Integration

This environment follows [AWS AIDLC](https://github.com/awslabs/aidlc-workflows) principles:
- **Adaptive difficulty**: 5 tasks from Easy to Expert, each requiring different strategies
- **Quality gates**: Partial reward at each step; warning detection as a final quality check
- **Structured audit logging**: AIDLC-compliant `[START]/[STEP]/[END]` stdout format
- **Content validation**: Agent edits are validated (max length, null bytes, Pydantic typing) before application
