"""
OpenEnv server entry point.

This module satisfies the openenv validate contract:
  - pyproject.toml [project.scripts] server = "server.app:main"
  - def main() callable with if __name__ == "__main__": main()

It mounts the core FastAPI application from env.py and adds the three
extra endpoints required by openenv validate --url (runtime validation):
  GET  /metadata  → environment name + description
  GET  /schema    → action / observation / state JSON schemas
  POST /mcp       → minimal JSON-RPC 2.0 stub (openenv protocol handshake)

The /openapi.json schema is always exposed (needed by runtime validator)
even though the interactive Swagger UI is only shown when DEBUG=true.
"""

import os
import sys

# Ensure the project root is importable when executed as `uv run server`
# (pyproject.toml scripts put CWD on sys.path, but be explicit).
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from env import (  # noqa: E402
    Action,
    Observation,
    Reward,
    State,
    TASKS,
    app,
)


# ---------------------------------------------------------------------------
# /metadata  (required by openenv validate --url)
# ---------------------------------------------------------------------------
@app.get("/metadata")
async def metadata():
    """Return environment name and description for the runtime validator."""
    return {
        "name": "rust-swe-agent-env",
        "description": (
            "RL environment where an AI agent fixes broken Rust projects "
            "by editing Cargo.toml and src/main.rs until cargo check succeeds."
        ),
        "version": "1.0.0",
        "tags": ["real-world", "swe", "rust", "ci-cd"],
        "tasks": len(TASKS),
    }


# ---------------------------------------------------------------------------
# /schema  (required by openenv validate --url)
# ---------------------------------------------------------------------------
@app.get("/schema")
async def schema():
    """Return JSON schemas for action, observation, and state models."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


# ---------------------------------------------------------------------------
# /mcp  (required by openenv validate --url — minimal JSON-RPC 2.0 stub)
# ---------------------------------------------------------------------------
@app.post("/mcp")
async def mcp():
    """
    Minimal MCP / JSON-RPC 2.0 handshake endpoint.
    openenv validate --url checks that POST /mcp returns {"jsonrpc": "2.0"}.
    """
    return {
        "jsonrpc": "2.0",
        "method": "initialize",
        "result": {
            "name": "rust-swe-agent-env",
            "version": "1.0.0",
            "capabilities": ["reset", "step", "state", "schema"],
        },
    }


# ---------------------------------------------------------------------------
# Entry point (required by pyproject.toml [project.scripts])
# ---------------------------------------------------------------------------
def main() -> None:
    """Start the OpenEnv server via uvicorn."""
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
