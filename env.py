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
from urllib.parse import urlparse

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# In-memory audit ring buffer (last 200 log lines surfaced at /logs)
import collections
_AUDIT_BUFFER: collections.deque = collections.deque(maxlen=200)

class _AuditHandler(logging.Handler):
    _fmt = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        _AUDIT_BUFFER.append({
            "ts": self._fmt.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        })

_audit_handler = _AuditHandler()
_audit_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger().addHandler(_audit_handler)

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
    """
    One agent action. Three modes:

    - file_to_edit = "Cargo.toml" | "src/main.rs"
        Classic single-file edit.  new_content required.
    - file_to_edit = "both_files"
        Atomic edit of both files in one step.
        cargo_toml_content + main_rs_content required.  new_content unused.
    - file_to_edit = "dry_run"
        Re-runs cargo check without touching any file.
        Useful for a planning step to refresh compiler diagnostics.
        No content fields required.  Reward is always 0.0 / not done.
    """
    model_config = ConfigDict(extra="forbid")  # A03 — reject unknown fields

    file_to_edit: Literal["Cargo.toml", "src/main.rs", "both_files", "dry_run"] = Field(
        ...,
        description="'Cargo.toml', 'src/main.rs', 'both_files', or 'dry_run'",
    )
    new_content: Optional[str] = Field(
        default=None,
        description="Complete new file content (required for single-file edits)",
        max_length=131_072,
    )
    cargo_toml_content: Optional[str] = Field(
        default=None,
        description="New Cargo.toml content (required when file_to_edit='both_files')",
        max_length=131_072,
    )
    main_rs_content: Optional[str] = Field(
        default=None,
        description="New src/main.rs content (required when file_to_edit='both_files')",
        max_length=131_072,
    )

    @field_validator("new_content", "cargo_toml_content", "main_rs_content", mode="before")
    @classmethod
    def no_null_bytes(cls, v: Optional[str]) -> Optional[str]:
        if v and "\x00" in v:
            raise ValueError("content must not contain null bytes")
        return v

    @model_validator(mode="after")
    def validate_fields_for_mode(self) -> "Action":
        fte = self.file_to_edit
        if fte in ("Cargo.toml", "src/main.rs"):
            if not self.new_content:
                raise ValueError(f"new_content is required when file_to_edit='{fte}'")
        elif fte == "both_files":
            if not self.cargo_toml_content or not self.main_rs_content:
                raise ValueError(
                    "cargo_toml_content and main_rs_content are both required "
                    "when file_to_edit='both_files'"
                )
        # dry_run: no content fields required
        return self


class Reward(BaseModel):
    # Validator requires strictly (0, 1) — gt/lt enforces this at the model level.
    score: float = Field(default=0.01, gt=0.0, lt=1.0)
    is_done: bool = Field(default=False)


class Info(BaseModel):
    """Structured metadata returned alongside each step (OpenEnv spec: info dict)."""
    error_count: int = Field(default=0, description="Current compiler error count")
    warning_count: int = Field(default=0, description="Current compiler warning count")
    initial_error_count: int = Field(default=0, description="Error count at episode start")
    errors_fixed: int = Field(default=0, description="Errors fixed since reset")
    regression: bool = Field(default=False, description="True if error count increased this step")
    is_dry_run: bool = Field(default=False, description="True if this step was a dry_run (no files changed)")
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


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="GitHub repository URL to analyze")


class AnalyzeResponse(BaseModel):
    repo_url: str
    cargo_toml: str = Field(default="")
    main_rs: str = Field(default="")
    compiler_output: str = Field(default="")
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    builds: bool = Field(default=False)
    files_found: list = Field(default_factory=list)


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

        is_dry_run = action.file_to_edit == "dry_run"

        if action.file_to_edit == "Cargo.toml":
            (ws / "Cargo.toml").write_text(action.new_content, encoding="utf-8")
        elif action.file_to_edit == "src/main.rs":
            src = ws / "src"
            src.mkdir(parents=True, exist_ok=True)
            (src / "main.rs").write_text(action.new_content, encoding="utf-8")
        elif action.file_to_edit == "both_files":
            src = ws / "src"
            src.mkdir(parents=True, exist_ok=True)
            (ws / "Cargo.toml").write_text(action.cargo_toml_content, encoding="utf-8")
            (src / "main.rs").write_text(action.main_rs_content, encoding="utf-8")
        # dry_run: write nothing, just re-run cargo check

        returncode, compiler_output = self._run_cargo_check()
        cargo_toml_content, main_rs_content = self._read_files()

        self._last_observation = Observation(
            compiler_output=compiler_output,
            cargo_toml_content=cargo_toml_content,
            main_rs_content=main_rs_content,
        )

        current_errors = self._count_errors(compiler_output)
        current_warnings = self._count_warnings(compiler_output)
        regression = (not is_dry_run) and (current_errors > self._previous_error_count)
        if not is_dry_run:
            self._previous_error_count = current_errors

        # --- Reward with partial progress ---
        # IMPORTANT: Validator requires scores strictly in (0, 1) — never 0.0 or 1.0.
        # Ladder: regression=0.01, dry_run=0.01, stalled=0.05, partial=0.09-0.90,
        #         build+warnings=0.91-0.98, clean build=0.99
        if is_dry_run:
            # Planning step — no file changes, minimal reward.
            self._last_reward = Reward(score=0.01, is_done=False)
        elif returncode == 0:
            # Build passed. Deduct for warnings; cap at 0.99 (never 1.0).
            score = 0.99 if current_warnings == 0 else max(0.91, 0.99 - current_warnings * 0.01)
            self._last_reward = Reward(score=round(score, 2), is_done=True)
        elif self._initial_error_count > 0:
            reduction = self._initial_error_count - current_errors
            if reduction > 0:
                ratio = reduction / self._initial_error_count
                # Range: 0.09–0.90 (never 0.0 or 1.0)
                self._last_reward = Reward(
                    score=round(max(0.09, min(0.90, ratio * 0.90)), 2), is_done=False
                )
            elif regression:
                # Agent made things worse — lowest possible signal.
                self._last_reward = Reward(score=0.01, is_done=False)
            else:
                # Stalled — slightly above regression floor.
                self._last_reward = Reward(score=0.05, is_done=False)
        else:
            # No initial errors recorded — shouldn't happen, safe floor.
            self._last_reward = Reward(score=0.01, is_done=False)

        # Build the info dict (OpenEnv spec compliance).
        self._last_info = Info(
            error_count=current_errors,
            warning_count=current_warnings,
            initial_error_count=self._initial_error_count,
            errors_fixed=max(0, self._initial_error_count - current_errors),
            regression=regression,
            is_dry_run=is_dry_run,
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
async def audit_middleware(request: Request, call_next):
    """Log every request with method, path, status, and wall-time."""
    import time
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "REQUEST method=%s path=%s status=%d elapsed_ms=%.1f remote=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request.client.host if request.client else "unknown",
    )
    return response


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
    """Interactive demo UI.  Falls back to JSON if the HTML file is missing."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
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


@app.get("/logs")
async def get_logs(n: int = 50):
    """Return the last N audit log entries (max 200). Useful for debugging."""
    n = min(n, 200)
    entries = list(_AUDIT_BUFFER)[-n:]
    return {"count": len(entries), "entries": entries}


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


def _clone_and_check(repo_url: str) -> AnalyzeResponse:
    """
    Clone a GitHub repo (depth=1), detect workspace vs single-crate,
    run the appropriate cargo check command, and return structured results.
    Runs in a thread-pool executor — never blocks the event loop.

    BUG FIXED: workspace repos (e.g. rustlings) now use `cargo check --workspace`
    so all member crates are checked, not just the (empty) root manifest.
    """
    import re
    import time

    audit_id = str(uuid.uuid4())[:8]
    t_start = time.monotonic()

    logger.info("ANALYZE_START audit_id=%s repo=%s", audit_id, repo_url)

    # A02 — validate URL is a GitHub repo (SSRF mitigation)
    parsed = urlparse(repo_url)
    if parsed.scheme not in ("http", "https"):
        logger.warning("ANALYZE_REJECT audit_id=%s reason=bad_scheme url=%s", audit_id, repo_url)
        raise ValueError("Only http/https URLs are allowed")
    if parsed.hostname not in ("github.com", "www.github.com"):
        logger.warning("ANALYZE_REJECT audit_id=%s reason=non_github host=%s", audit_id, parsed.hostname)
        raise ValueError("Only github.com repositories are supported")

    with tempfile.TemporaryDirectory(prefix="oxidizer_analyze_") as tmpdir:
        clone_dir = Path(tmpdir) / "repo"

        # ── Clone ──────────────────────────────────────────────────────
        t_clone = time.monotonic()
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(clone_dir)],
                capture_output=True, text=True, timeout=60, check=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("ANALYZE_CLONE_TIMEOUT audit_id=%s elapsed=%.1fs", audit_id, time.monotonic() - t_clone)
            raise ValueError("Repository clone timed out (60s limit)")
        except subprocess.CalledProcessError as exc:
            logger.error("ANALYZE_CLONE_FAIL audit_id=%s stderr=%s", audit_id, exc.stderr[:200])
            raise ValueError(f"Clone failed: {exc.stderr[:200]}")

        logger.info("ANALYZE_CLONED audit_id=%s elapsed=%.1fs", audit_id, time.monotonic() - t_clone)

        # ── Detect project structure ───────────────────────────────────
        cargo_path = clone_dir / "Cargo.toml"
        if not cargo_path.exists():
            logger.warning("ANALYZE_NO_CARGO_TOML audit_id=%s", audit_id)
            return AnalyzeResponse(
                repo_url=repo_url,
                compiler_output="No Cargo.toml found in repository root. Is this a Rust project?",
                files_found=[],
            )

        cargo_toml_text = cargo_path.read_text(encoding="utf-8", errors="replace")[:65536]

        # Key fix: detect Cargo workspace to avoid false "builds successfully"
        is_workspace = "[workspace]" in cargo_toml_text
        cargo_cmd = ["cargo", "check", "--workspace"] if is_workspace else ["cargo", "check"]

        logger.info(
            "ANALYZE_PROJECT audit_id=%s is_workspace=%s cmd=%s",
            audit_id, is_workspace, " ".join(cargo_cmd),
        )

        # Collect files found for response
        files_found = ["Cargo.toml"]
        src_main_path = clone_dir / "src" / "main.rs"
        src_lib_path = clone_dir / "src" / "lib.rs"

        main_rs_text = ""
        if src_main_path.exists():
            files_found.append("src/main.rs")
            main_rs_text = src_main_path.read_text(encoding="utf-8", errors="replace")[:65536]
        elif src_lib_path.exists():
            files_found.append("src/lib.rs")
            main_rs_text = src_lib_path.read_text(encoding="utf-8", errors="replace")[:65536]

        if is_workspace:
            # For workspaces, show the workspace Cargo.toml and a note
            main_rs_text = (
                "# Workspace repository — showing root Cargo.toml members.\n"
                "# Individual crate sources are in subdirectories.\n"
            )
            # Collect member directories for context
            members = re.findall(r'members\s*=\s*\[([^\]]*)\]', cargo_toml_text, re.DOTALL)
            if members:
                member_list = re.findall(r'"([^"]+)"', members[0])
                main_rs_text += f"# Members ({len(member_list)}): " + ", ".join(member_list[:10])
                if len(member_list) > 10:
                    main_rs_text += f" ... and {len(member_list)-10} more"

        # ── Run cargo check ────────────────────────────────────────────
        t_check = time.monotonic()
        env = os.environ.copy()
        env["CARGO_TERM_COLOR"] = "never"
        env["RUST_BACKTRACE"] = "0"

        returncode = 1
        try:
            result = subprocess.run(
                cargo_cmd,
                cwd=str(clone_dir),
                capture_output=True, text=True, timeout=120, check=False, env=env,
            )
            compiler_output = f"{result.stdout}\n{result.stderr}".strip()
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            compiler_output = "cargo check timed out after 120s"
            logger.error("ANALYZE_CHECK_TIMEOUT audit_id=%s elapsed=%.1fs", audit_id, time.monotonic() - t_check)
        except Exception as exc:
            compiler_output = f"cargo check error: {type(exc).__name__}"
            logger.error("ANALYZE_CHECK_ERROR audit_id=%s type=%s", audit_id, type(exc).__name__)

        error_count = RustFixerEnv._count_errors(compiler_output)
        warning_count = RustFixerEnv._count_warnings(compiler_output)
        builds = (returncode == 0)

        logger.info(
            "ANALYZE_DONE audit_id=%s repo=%s workspace=%s builds=%s "
            "errors=%d warnings=%d check_elapsed=%.1fs total_elapsed=%.1fs",
            audit_id, repo_url, is_workspace, builds,
            error_count, warning_count,
            time.monotonic() - t_check,
            time.monotonic() - t_start,
        )

        return AnalyzeResponse(
            repo_url=repo_url,
            cargo_toml=cargo_toml_text,
            main_rs=main_rs_text,
            compiler_output=compiler_output[:8192],
            error_count=error_count,
            warning_count=warning_count,
            builds=builds,
            files_found=files_found,
        )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(request: AnalyzeRequest):
    """Clone a GitHub Rust repo, run cargo check (--workspace for workspaces), return analysis."""
    try:
        async with _CARGO_SEMAPHORE:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _clone_and_check, request.repo_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise _opaque_error(exc, "analyze failed") from exc


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


_MCP_TOOLS = [
    {
        "name": "get_compiler_errors",
        "description": (
            "Parse the current compiler output and return a structured list of "
            "error codes and messages. Useful for planning which files to fix."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_task_state",
        "description": (
            "Return the current environment state: task name, error counts, "
            "step count, and latest compiler output."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_tasks",
        "description": "Return descriptions of all 5 available tasks with their difficulty levels.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]


def _mcp_error(req_id, code: int, message: str):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


@app.post("/mcp")
async def mcp(request: Request):
    """
    JSON-RPC 2.0 endpoint.  Supports:
      initialize    — capability handshake (required by openenv validate)
      tools/list    — enumerate available agent tools
      tools/call    — invoke a tool by name
    """
    try:
        body = await request.json()
    except Exception:
        return _mcp_error(None, -32700, "Parse error")

    req_id = body.get("id")
    method = body.get("method", "")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "name": "rust-swe-agent-env",
                "version": "1.0.0",
                "capabilities": {
                    "tools": True,
                    "actions": ["Cargo.toml", "src/main.rs", "both_files", "dry_run"],
                },
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _MCP_TOOLS}}

    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")

        if tool_name == "get_compiler_errors":
            import re as _re
            async with _ENV_LOCK:
                env = get_env()
                if env._last_observation is None:
                    return _mcp_error(req_id, -32002, "Call reset() first")
                raw = env._last_observation.compiler_output
                errors = _re.findall(r"error\[([A-Z]\d+)\].*", raw)
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "error_count": env._last_info.error_count,
                        "error_codes": errors,
                        "compiler_output": raw,
                    },
                }

        if tool_name == "get_task_state":
            async with _ENV_LOCK:
                try:
                    state = get_env().get_state()
                except RuntimeError as exc:
                    return _mcp_error(req_id, -32002, str(exc))
            return {"jsonrpc": "2.0", "id": req_id, "result": state.model_dump()}

        if tool_name == "list_tasks":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "tasks": [
                        {"id": i, "name": t.name, "description": t.description}
                        for i, t in enumerate(TASKS)
                    ]
                },
            }

        return _mcp_error(req_id, -32601, f"Unknown tool: {tool_name!r}")

    return _mcp_error(req_id, -32601, f"Method not found: {method!r}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("env:app", host="0.0.0.0", port=7860, log_level="info")
