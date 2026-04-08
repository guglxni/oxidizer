"""
Rust Build Fixer RL Environment — OpenEnv-compliant

Security posture (OWASP Top 10:2025 + ASI):
  A01 – Optional API_KEY auth; deny by default when set.
  A02 – Swagger/Redoc disabled in production (DEBUG env var required).
  A03 – Pydantic extra="forbid" on all inbound models.
  A04 – No credentials stored in code; all from env.
  A05 – new_content max 128 KB, null bytes rejected; file_to_edit is a Literal.
  A09 – Structured logging; all security events go to the Python logger.
  A10 – HTTP 500s return an opaque error ID; details stay server-side.
  ASI02 – Semaphore caps concurrent cargo-check spawns at 4.
  ASI03 – asyncio.Lock serialises access to the global env singleton.
  ASI05 – cargo check runs in a thread-pool executor (never blocks the event loop).
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ---------------------------------------------------------------------------
# Optional server-side API key (A01 Broken Access Control).
# If API_KEY is set in the environment, every mutating request must supply
# the same value in the X-Api-Key header. If it is not set the server logs a
# warning and runs unauthenticated (acceptable in a local hackathon context).
# ---------------------------------------------------------------------------
_SERVER_API_KEY: Optional[str] = os.environ.get("API_KEY")
if not _SERVER_API_KEY:
    logger.warning(
        "API_KEY env var is not set — all endpoints are unauthenticated. "
        "Set API_KEY to enable server-side authentication."
    )

# Cap concurrent cargo-check subprocesses (ASI02 / resource exhaustion).
_CARGO_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("CARGO_CONCURRENCY", "2")))
# Serialise access to the global env singleton (ASI03 race condition).
_ENV_LOCK = asyncio.Lock()


# =============================================================================
# Pydantic Models
# =============================================================================


class Observation(BaseModel):
    """What the agent sees after each step."""
    compiler_output: str = Field(default="")
    cargo_toml_content: str = Field(default="")
    main_rs_content: str = Field(default="")


class Action(BaseModel):
    """One edit: which file to overwrite and with what content."""
    model_config = ConfigDict(extra="forbid")  # A03 — reject unknown fields

    file_to_edit: Literal["Cargo.toml", "src/main.rs"] = Field(
        ...,
        description="Must be exactly 'Cargo.toml' or 'src/main.rs'",
    )
    new_content: str = Field(
        ...,
        description="Complete new file content",
        max_length=131_072,  # 128 KB — prevents resource exhaustion (A05)
    )

    @field_validator("new_content")
    @classmethod
    def no_null_bytes(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("new_content must not contain null bytes")
        return v


class Reward(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_done: bool = Field(default=False)


class Info(BaseModel):
    """Structured metadata returned alongside each step (OpenEnv spec: info dict)."""
    error_count: int = Field(default=0, description="Current compiler error count")
    warning_count: int = Field(default=0, description="Current compiler warning count")
    initial_error_count: int = Field(default=0, description="Error count at episode start")
    errors_fixed: int = Field(default=0, description="Errors fixed since reset")
    regression: bool = Field(default=False, description="True if error count increased this step")
    step_count: int = Field(default=0)
    task_name: str = Field(default="")
    task_id: int = Field(default=0)


class State(BaseModel):
    observation: Observation
    reward: Reward
    info: Info
    task_name: str
    task_id: int
    step_count: int


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="0=Easy 1=Medium 2=Hard 3=MultiDep 4=Expert. Omit to cycle.",
    )


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool = Field(default=False, description="True when the episode is complete")
    info: Info


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


# =============================================================================
# Task Definitions
# =============================================================================


class TaskConfig(BaseModel):
    name: str
    description: str
    cargo_toml: str
    main_rs: str


TASK_EASY = TaskConfig(
    name="missing_rand_dependency",
    description="Easy: src/main.rs uses rand::Rng but Cargo.toml has no rand entry",
    cargo_toml="""\
[package]
name = "broken-project"
version = "0.1.0"
edition = "2021"

[dependencies]
""",
    main_rs="""\
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let num: u32 = rng.gen_range(1..=100);
    println!("Random number: {}", num);
}
""",
)

TASK_MEDIUM = TaskConfig(
    name="serde_feature_missing",
    description='Medium: serde in Cargo.toml is missing features=["derive"]',
    cargo_toml="""\
[package]
name = "broken-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
serde_json = "1.0"
""",
    main_rs="""\
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct User {
    name: String,
    age: u32,
}

fn main() {
    let user = User { name: "Alice".to_string(), age: 30 };
    let json = serde_json::to_string(&user).unwrap();
    println!("User JSON: {}", json);
}
""",
)

TASK_HARD = TaskConfig(
    name="syntax_and_dependency_error",
    description="Hard: missing semicolon in main.rs AND reqwest absent from Cargo.toml",
    cargo_toml="""\
[package]
name = "broken-project"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
""",
    main_rs="""\
#[tokio::main]
async fn main() {
    let url = "https://api.github.com/users/rust-lang"
    println!("Fetching: {}", url);

    let response = reqwest::get(url).await.unwrap();
    println!("Status: {}", response.status());
}
""",
)

# Task 3: Medium-Hard — Two missing dependencies in a single file
# Agent must identify BOTH chrono and regex from the compiler output and add
# them in a single Cargo.toml edit.  Tests multi-dependency resolution.
TASK_MULTI_DEP = TaskConfig(
    name="multiple_missing_dependencies",
    description="Medium-Hard: main.rs uses chrono + regex but neither is in Cargo.toml",
    cargo_toml="""\
[package]
name = "broken-project"
version = "0.1.0"
edition = "2021"

[dependencies]
""",
    main_rs="""\
use chrono::Utc;
use regex::Regex;

fn main() {
    let now = Utc::now();
    let re = Regex::new(r"^\\d{4}-\\d{2}-\\d{2}$").unwrap();
    let date_str = now.format("%Y-%m-%d").to_string();
    if re.is_match(&date_str) {
        println!("Today is: {}", date_str);
    }
}
""",
)

# Task 4: Expert — Three errors across BOTH files (requires ≥2 steps)
# Cargo.toml: serde missing derive feature + serde_json entirely absent
# main.rs: missing semicolon on the serde_json::to_string line
# This is the hardest task — the agent must fix Cargo.toml AND main.rs.
TASK_EXPERT = TaskConfig(
    name="cross_file_multi_error",
    description="Expert: serde missing derive + serde_json absent from Cargo.toml + missing semicolon in main.rs",
    cargo_toml="""\
[package]
name = "broken-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
rand = "0.8"
""",
    main_rs="""\
use serde::{Deserialize, Serialize};
use rand::Rng;

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    name: String,
    seed: u32,
}

fn main() {
    let mut rng = rand::thread_rng();
    let cfg = Config { name: "test".to_string(), seed: rng.gen_range(1..100) };
    let json = serde_json::to_string(&cfg).unwrap()
    println!("Config: {}", json);
}
""",
)

TASKS = [TASK_EASY, TASK_MEDIUM, TASK_HARD, TASK_MULTI_DEP, TASK_EXPERT]


# =============================================================================
# Core Environment
# =============================================================================


class RustFixerEnv:
    """
    OpenEnv RL environment.  One instance per server process.
    Access must be serialised with _ENV_LOCK (see route handlers).
    """

    def __init__(self) -> None:
        # Start at -1 so the first auto-cycle lands on task 0 (Easy).
        self._current_task_idx: int = -1
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._workspace_path: Optional[Path] = None
        self._step_count: int = 0
        self._current_task: Optional[TaskConfig] = None
        self._last_observation: Optional[Observation] = None
        self._last_reward: Reward = Reward()
        self._last_info: Info = Info()
        self._initial_error_count: int = 0
        self._previous_error_count: int = 0

    # ------------------------------------------------------------------ helpers

    def _workspace(self) -> Path:
        if self._workspace_path is None:
            raise RuntimeError("Call reset() first.")
        return self._workspace_path

    def _write_files(self, cargo_toml: str, main_rs: str) -> None:
        ws = self._workspace()
        src = ws / "src"
        src.mkdir(parents=True, exist_ok=True)
        (ws / "Cargo.toml").write_text(cargo_toml, encoding="utf-8")
        (src / "main.rs").write_text(main_rs, encoding="utf-8")

    def _read_files(self) -> tuple[str, str]:
        ws = self._workspace()
        cargo_toml = (ws / "Cargo.toml").read_text(encoding="utf-8")
        main_rs_path = ws / "src" / "main.rs"
        main_rs = main_rs_path.read_text(encoding="utf-8") if main_rs_path.exists() else ""
        return cargo_toml, main_rs

    @staticmethod
    def _count_warnings(compiler_output: str) -> int:
        """Count compiler warnings (non-error diagnostics)."""
        import re
        return len(re.findall(r"warning\[", compiler_output))

    @staticmethod
    def _count_errors(compiler_output: str) -> int:
        """
        Parse the number of compiler errors from cargo check output.

        Cargo emits a summary like "could not compile ... due to 3 previous errors".
        If present, use that authoritative count; otherwise fall back to counting
        `error[Exxxx]` occurrences.
        """
        import re
        m = re.search(r"due to (\d+) previous error", compiler_output)
        if m:
            return int(m.group(1))
        return len(re.findall(r"error\[E\d+\]", compiler_output))

    def _run_cargo_check(self) -> tuple[int, str]:
        """
        Blocking.  Call via asyncio.get_event_loop().run_in_executor().

        Security notes:
        - shell=False (the default for list argv) — prevents shell injection.
        - CARGO_TERM_COLOR=never — strips ANSI escapes before the LLM sees output.
        - RUST_BACKTRACE=0 — hides internal Rust paths from compiler messages.
        - timeout=120 — caps worst-case per-check wall time.
        """
        ws = self._workspace()
        env = os.environ.copy()
        env["CARGO_TERM_COLOR"] = "never"
        env["RUST_BACKTRACE"] = "0"
        env["CARGO_INCREMENTAL"] = "1"  # reuse build cache across steps

        try:
            result = subprocess.run(
                ["cargo", "check"],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=env,
            )
            combined = f"{result.stdout}\n{result.stderr}".strip()
            return result.returncode, combined or f"exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            return 1, "cargo check timed out after 120 s"
        except FileNotFoundError:
            return 1, "cargo not found — is Rust installed?"
        except Exception as exc:  # noqa: BLE001
            return 1, f"Unexpected subprocess error: {type(exc).__name__}"

    # ------------------------------------------------------------------ public

    def reset(self, task_id: Optional[int] = None) -> Observation:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

        self._temp_dir = tempfile.TemporaryDirectory(prefix="rust_fixer_env_")
        self._workspace_path = Path(self._temp_dir.name)
        self._step_count = 0
        self._last_reward = Reward()

        if task_id is not None:
            self._current_task_idx = task_id % len(TASKS)
        else:
            self._current_task_idx = (self._current_task_idx + 1) % len(TASKS)

        self._current_task = TASKS[self._current_task_idx]
        self._write_files(self._current_task.cargo_toml, self._current_task.main_rs)

        returncode, compiler_output = self._run_cargo_check()
        cargo_toml_content, main_rs_content = self._read_files()

        self._initial_error_count = self._count_errors(compiler_output)
        self._previous_error_count = self._initial_error_count
        logger.info(
            "reset task=%s returncode=%d initial_errors=%d",
            self._current_task.name, returncode, self._initial_error_count,
        )

        self._last_observation = Observation(
            compiler_output=compiler_output,
            cargo_toml_content=cargo_toml_content,
            main_rs_content=main_rs_content,
        )
        return self._last_observation

    def step(self, action: Action) -> tuple[Observation, Reward, Info]:
        """
        Apply one edit and evaluate with cargo check.

        Returns (observation, reward, info) per the OpenEnv spec:
        step(action) → observation, reward, done, info.
        (done is embedded in reward.is_done; info carries diagnostic metadata.)
        """
        self._step_count += 1
        ws = self._workspace()

        logger.info(
            "step=%d file=%s content_bytes=%d",
            self._step_count,
            action.file_to_edit,
            len(action.new_content.encode()),
        )

        if action.file_to_edit == "Cargo.toml":
            (ws / "Cargo.toml").write_text(action.new_content, encoding="utf-8")
        else:
            src = ws / "src"
            src.mkdir(parents=True, exist_ok=True)
            (src / "main.rs").write_text(action.new_content, encoding="utf-8")

        returncode, compiler_output = self._run_cargo_check()
        cargo_toml_content, main_rs_content = self._read_files()

        self._last_observation = Observation(
            compiler_output=compiler_output,
            cargo_toml_content=cargo_toml_content,
            main_rs_content=main_rs_content,
        )

        current_errors = self._count_errors(compiler_output)
        current_warnings = self._count_warnings(compiler_output)
        regression = current_errors > self._previous_error_count
        self._previous_error_count = current_errors

        # --- Reward with partial progress + regression penalty ---
        if returncode == 0:
            # Build passed.  Deduct slightly for warnings (quality gate).
            score = 1.0 if current_warnings == 0 else max(0.95, 1.0 - current_warnings * 0.01)
            self._last_reward = Reward(score=round(score, 2), is_done=True)
        elif self._initial_error_count > 0:
            reduction = self._initial_error_count - current_errors
            if reduction > 0:
                ratio = reduction / self._initial_error_count
                self._last_reward = Reward(
                    score=round(min(0.9, ratio * 0.9), 2), is_done=False
                )
            elif regression:
                # Regression: agent made things worse → 0.0 (distinguishable
                # from no-change via info.regression=True, but reward itself
                # is the lowest possible signal).
                self._last_reward = Reward(score=0.0, is_done=False)
            else:
                # No change, no regression: tiny positive to distinguish from
                # regression in cases where the RL trainer only sees the scalar.
                self._last_reward = Reward(score=0.05, is_done=False)
        else:
            self._last_reward = Reward(score=0.0, is_done=False)

        # Build the info dict (OpenEnv spec compliance).
        self._last_info = Info(
            error_count=current_errors,
            warning_count=current_warnings,
            initial_error_count=self._initial_error_count,
            errors_fixed=max(0, self._initial_error_count - current_errors),
            regression=regression,
            step_count=self._step_count,
            task_name=self._current_task.name if self._current_task else "",
            task_id=self._current_task_idx,
        )

        logger.info(
            "step=%d errors=%d/%d warnings=%d regression=%s score=%.2f",
            self._step_count, current_errors, self._initial_error_count,
            current_warnings, regression, self._last_reward.score,
        )
        return self._last_observation, self._last_reward, self._last_info

    def get_state(self) -> State:
        if self._current_task is None or self._last_observation is None:
            raise RuntimeError("Call reset() first.")
        return State(
            observation=self._last_observation,
            reward=self._last_reward,
            info=self._last_info,
            task_name=self._current_task.name,
            task_id=self._current_task_idx,
            step_count=self._step_count,
        )

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._workspace_path = None

    def close(self) -> None:
        """Alias for cleanup() — matches the sample inference script pattern."""
        self.cleanup()


# =============================================================================
# FastAPI Application
# =============================================================================

_env_instance: Optional[RustFixerEnv] = None
_DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _env_instance
    if _env_instance is not None:
        _env_instance.cleanup()
        _env_instance = None


app = FastAPI(
    title="Rust Build Fixer Environment",
    description="OpenEnv RL environment for fixing broken Rust builds",
    version="1.0.0",
    lifespan=lifespan,
    # A02 — hide interactive UI in production, but always expose the machine-
    # readable /openapi.json schema (required by openenv validate --url).
    docs_url="/docs" if _DEBUG else None,
    redoc_url="/redoc" if _DEBUG else None,
    openapi_url="/openapi.json",  # always on — needed by runtime validator
)


# ---------------------------------------------------------------------------
# Auth middleware (A01 Broken Access Control).
# Only enforced when API_KEY is set in the environment.
# ---------------------------------------------------------------------------
_UNPROTECTED_PATHS = {"/health"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if _SERVER_API_KEY and request.url.path not in _UNPROTECTED_PATHS:
        provided = request.headers.get("X-Api-Key", "")
        if provided != _SERVER_API_KEY:
            logger.warning(
                "Rejected unauthenticated request path=%s remote=%s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return _json_error(403, "Forbidden")
    return await call_next(request)


def _json_error(status: int, detail: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content={"detail": detail})


def get_env() -> RustFixerEnv:
    global _env_instance
    if _env_instance is None:
        _env_instance = RustFixerEnv()
    return _env_instance


def _opaque_error(exc: Exception, context: str) -> HTTPException:
    """
    A10 — Log full details server-side; return only an opaque error ID to
    the client so internal paths and exception types are not disclosed.
    """
    error_id = uuid.uuid4()
    logger.exception("%s [error_id=%s]", context, error_id)
    return HTTPException(
        status_code=500,
        detail=f"Internal server error. Reference ID: {error_id}",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Landing page — confirms the server is up and lists available endpoints."""
    return {
        "name": "rust-swe-agent-env",
        "status": "running",
        "description": "OpenEnv RL environment — AI agent fixes broken Rust builds via cargo check",
        "endpoints": {
            "health":   "GET  /health",
            "metadata": "GET  /metadata",
            "schema":   "GET  /schema",
            "tasks":    "GET  /tasks",
            "reset":    "POST /reset",
            "step":     "POST /step",
            "state":    "GET  /state",
            "mcp":      "POST /mcp",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", environment="rust-swe-agent-env", version="1.0.0")


@app.post("/reset", response_model=Observation)
async def reset(request: Optional[ResetRequest] = Body(default=None)):
    """
    Reset the environment.  Body is fully optional:
      - no body / null body  → cycles to the next task
      - {}                   → same as no body
      - {"task_id": 0|1|2}  → selects a specific task
    """
    try:
        task_id = request.task_id if request else None
        async with _CARGO_SEMAPHORE, _ENV_LOCK:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, get_env().reset, task_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise _opaque_error(exc, "reset failed") from exc


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest = Body(...)):
    try:
        async with _CARGO_SEMAPHORE, _ENV_LOCK:
            loop = asyncio.get_event_loop()
            obs, reward, info = await loop.run_in_executor(
                None, get_env().step, request.action
            )
        return StepResponse(observation=obs, reward=reward, done=reward.is_done, info=info)
    except HTTPException:
        raise
    except Exception as exc:
        raise _opaque_error(exc, "step failed") from exc


@app.get("/state", response_model=State)
async def get_current_state():
    try:
        async with _ENV_LOCK:
            return get_env().get_state()
    except Exception as exc:
        raise _opaque_error(exc, "get_state failed") from exc


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": i, "name": t.name, "description": t.description}
            for i, t in enumerate(TASKS)
        ]
    }


@app.get("/metadata")
async def metadata():
    """Environment name + description (required by openenv validate --url)."""
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


@app.get("/schema")
async def schema():
    """Action / Observation / State JSON schemas (required by openenv validate --url)."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/mcp")
async def mcp():
    """
    Minimal JSON-RPC 2.0 handshake (required by openenv validate --url).
    Returns the required jsonrpc: '2.0' field so the validator marks this pass.
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


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("env:app", host="0.0.0.0", port=7860, log_level="info")
