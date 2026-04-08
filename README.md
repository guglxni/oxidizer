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

## Tasks

| ID | Difficulty | Error |
|----|------------|-------|
| 0 | Easy | `rand` crate missing from `Cargo.toml` |
| 1 | Medium | `serde` missing `features = ["derive"]` |
| 2 | Hard | Missing semicolon **+** `reqwest` absent from `Cargo.toml` |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/metadata` | GET | Environment name + description |
| `/schema` | GET | Action / Observation / State JSON schemas |
| `/reset` | POST | Reset to a task (body: `{"task_id": 0\|1\|2}`) |
| `/step` | POST | Apply one file edit (body: `{"action": {...}}`) |
| `/state` | GET | Current environment state |
| `/mcp` | POST | JSON-RPC 2.0 handshake |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes (inference) | LLM API base URL |
| `MODEL_NAME` | Yes (inference) | Model identifier |
| `HF_TOKEN` | Yes (inference) | Hugging Face / LLM API token |
| `API_KEY` | No | Server-side auth key (recommended for production) |
| `DEBUG` | No | Set to `true` to enable Swagger UI at `/docs` |
