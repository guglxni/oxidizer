"""
Baseline RL Agent for Rust Build Fixer Environment

Environment variables (per submission checklist):
  API_BASE_URL    – LLM API base URL  (default: https://api.openai.com/v1)
  MODEL_NAME      – model identifier  (default: gpt-4o)
  HF_TOKEN        – API token         (no default — must be set at runtime)
  LOCAL_IMAGE_NAME – optional Docker image for local testing

Security posture (OWASP Top 10:2025 + ASI):
  A02 – API_BASE_URL soft-validated (warn, no hard exit); A04 credentials from
        env only, never logged; A05 compiler output sanitised before LLM embed
        (ASI04 indirect prompt injection); A09 security events to stderr only;
        A10 LLM errors surface exception type, never raw message / stack trace.

AIDLC stdout log format (CI-validated, no extra lines, no embedded newlines):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import logging
import os
import re
import sys
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from openai import OpenAI

from env import Action, Observation, Reward, RustFixerEnv

# ---------------------------------------------------------------------------
# Logging — stderr only, never touches the CI stdout log stream.
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (A04 — credentials from env, never hardcoded)
#
# Per submission checklist:
#   - API_BASE_URL and MODEL_NAME have sensible defaults.
#   - HF_TOKEN has NO default; absence is a runtime warning, not a hard exit.
#   - LOCAL_IMAGE_NAME is optional (used with from_docker_image()).
# =============================================================================

# Match the exact sample inference.py variable pattern:
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = API_KEY  # alias for backward compat

BENCHMARK = os.getenv("RUST_FIXER_BENCHMARK", "rust-swe-agent-env")
MAX_STEPS = 10

# Maximum compiler output characters forwarded to the LLM (ASI04).
_MAX_COMPILER_OUTPUT_CHARS: int = 4_096


def _warn_config() -> None:
    """
    Soft-validate configuration at startup.

    A02 — log warnings for misconfiguration rather than calling sys.exit().
    The LLM call will surface a real error if credentials are wrong; a hard
    exit here would hide that signal from the CI log stream.
    """
    if not API_KEY:
        logger.warning(
            "Neither HF_TOKEN nor API_KEY is set. LLM calls will fail. "
            "Set HF_TOKEN or API_KEY before running inference.py."
        )

    # Soft-validate the URL (SSRF mitigation — warn, don't exit).
    try:
        parsed = urlparse(API_BASE_URL)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"bad URL: {API_BASE_URL!r}")
    except Exception as exc:
        logger.warning("API_BASE_URL may be invalid: %s", exc)

    if LOCAL_IMAGE_NAME:
        logger.info("LOCAL_IMAGE_NAME=%s (local Docker mode)", LOCAL_IMAGE_NAME)


_warn_config()


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """\
You are a Rust compilation-error fixing agent operating inside an RL loop.

Each turn you receive a JSON object with three keys:
  "cargo_toml"       – current Cargo.toml content
  "main_rs"          – current src/main.rs content
  "compiler_output"  – output of `cargo check --color never`

You must reply with ONLY a JSON object — no prose, no markdown fences:
{"file_to_edit": "Cargo.toml" | "src/main.rs", "new_content": "<full file content>"}

Rules:
1. Edit exactly ONE file per turn.
2. Provide the COMPLETE new content of that file.
3. Properly JSON-escape all backslashes, quotes, and newlines in new_content.
4. Read the compiler error codes (E0432, E0412, E0425 …) before deciding.

Common patterns:
- E0432 / "unresolved import"  → add the crate to [dependencies] in Cargo.toml
- E0277 / "Serialize not implemented" with serde → add features=["derive"]
- E0001 / "unexpected token" after a let binding → add the missing semicolon\
"""


# =============================================================================
# Compiler output sanitiser (ASI04 — indirect prompt injection defence)
# =============================================================================


def _sanitise_compiler_output(raw: str) -> str:
    """
    Strip content that could carry prompt-injection payloads before embedding
    compiler output in the LLM user message.

    Removes ANSI codes, C0 control characters (except \\t and \\n), and
    truncates to _MAX_COMPILER_OUTPUT_CHARS.
    """
    sanitised = re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", raw)
    sanitised = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", sanitised)
    if len(sanitised) > _MAX_COMPILER_OUTPUT_CHARS:
        sanitised = sanitised[:_MAX_COMPILER_OUTPUT_CHARS] + "\n[... truncated ...]"
    return sanitised


# =============================================================================
# LLM Client
# =============================================================================


class LLMClient:
    """
    Thin wrapper around the OpenAI-compatible client.

    All three credentials (API_BASE_URL, MODEL_NAME, HF_TOKEN) are injected
    at construction time so no global state is accessed inside methods.
    """

    def __init__(self, base_url: str, model: str, token: Optional[str]) -> None:
        self.model = model
        # A04 — if token is None, pass empty string; the API will return 401
        # and that error is surfaced cleanly in get_action().
        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=token or "",
        )

    def get_action(self, observation: Observation) -> tuple[Optional[Action], Optional[str]]:
        """
        Returns (Action, None) on success, (None, error_str) on failure.

        A10 — only the exception *type* is surfaced, never the message (which
        may contain the API key or server internals).
        """
        # ASI04 — wrap data in a JSON structure so compiler output is
        # unambiguously data, not instruction.
        user_data = {
            "cargo_toml": observation.cargo_toml_content,
            "main_rs": observation.main_rs_content,
            "compiler_output": _sanitise_compiler_output(observation.compiler_output),
        }
        user_prompt = (
            "Analyse this Rust project and return the JSON fix object:\n"
            + json.dumps(user_data, ensure_ascii=False)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content or ""
            action = _parse_action(raw)
            if action is None:
                safe_excerpt = raw[:120].replace("\n", " ")
                logger.warning("Unparseable LLM response: %s", safe_excerpt)
                return None, f"unparseable_response:{safe_excerpt[:80]}"
            return action, None

        except Exception as exc:  # noqa: BLE001
            # Surface type only — not message (A10 / A04).
            err = type(exc).__name__
            logger.error("LLM API call failed: %s: %s", err, exc)
            return None, err


def _parse_action(content: str) -> Optional[Action]:
    """
    Extract a valid Action from LLM output.

    Tries (in order):
    1. json.loads on content of a ```json … ``` fence.
    2. json.loads on the first top-level JSON object in bare text.
    3. json.loads on the entire stripped response.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidates: List[str] = []
    if fence_match:
        candidates.append(fence_match.group(1))

    bare_match = re.search(
        r'\{[^{}]*"file_to_edit"[^{}]*"new_content"[^{}]*\}',
        content,
        re.DOTALL,
    )
    if bare_match:
        candidates.append(bare_match.group(0))

    candidates.append(content.strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            fte = data.get("file_to_edit", "")
            nc = data.get("new_content", "")
            if fte in ("Cargo.toml", "src/main.rs") and nc:
                return Action(file_to_edit=fte, new_content=nc)
        except (json.JSONDecodeError, ValueError):
            continue

    return None


# =============================================================================
# Structured Logger — AIDLC-compliant stdout
# =============================================================================


class StructuredLogger:
    """
    All output goes to stdout via sys.stdout.write (never print()) so it is
    never interleaved with Python logging output on stderr.
    """

    def __init__(self, task_name: str, benchmark: str, model_name: str) -> None:
        self.task_name = task_name
        self.benchmark = benchmark
        self.model_name = model_name
        self.rewards: List[str] = []
        self.step_count: int = 0

    @staticmethod
    def _tok(value: str) -> str:
        """Collapse whitespace (including newlines) to underscore — keeps log lines single-line."""
        return re.sub(r"\s+", "_", value)[:120]

    def log_start(self) -> None:
        sys.stdout.write(
            f"[START] task={self.task_name} env={self.benchmark} model={self.model_name}\n"
        )
        sys.stdout.flush()

    def log_step(
        self,
        step: int,
        action: Action,
        reward: Reward,
        error: Optional[str] = None,
    ) -> None:
        self.step_count = step
        self.rewards.append(f"{reward.score:.2f}")
        sys.stdout.write(
            f"[STEP] step={step}"
            f" action={self._tok('edit:' + action.file_to_edit)}"
            f" reward={reward.score:.2f}"
            f" done={'true' if reward.is_done else 'false'}"
            f" error={'null' if error is None else self._tok(error)}\n"
        )
        sys.stdout.flush()

    def log_step_error(self, step: int, error_msg: str) -> None:
        self.step_count = step
        self.rewards.append("0.00")
        sys.stdout.write(
            f"[STEP] step={step} action=parse_error"
            f" reward=0.00 done=false error={self._tok(error_msg)}\n"
        )
        sys.stdout.flush()

    def log_end(self, success: bool, final_score: float) -> None:
        sys.stdout.write(
            f"[END] success={str(success).lower()}"
            f" steps={self.step_count}"
            f" score={final_score:.2f}"
            f" rewards={','.join(self.rewards)}\n"
        )
        sys.stdout.flush()


# =============================================================================
# Agent loop
# =============================================================================


def run_agent(task_id: int, benchmark: str = BENCHMARK) -> Tuple[bool, int, float]:
    task_names = [
        "missing_rand_dependency",
        "serde_feature_missing",
        "syntax_and_dependency_error",
        "multiple_missing_dependencies",
        "cross_file_multi_error",
    ]
    task_name = task_names[task_id]

    env = RustFixerEnv()
    observation = env.reset(task_id=task_id)

    logger.info("Agent starting task=%s model=%s", task_name, MODEL_NAME)

    slog = StructuredLogger(task_name=task_name, benchmark=benchmark, model_name=MODEL_NAME)
    llm = LLMClient(base_url=API_BASE_URL, model=MODEL_NAME, token=API_KEY)

    slog.log_start()

    success = False
    step = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            action, err = llm.get_action(observation)

            if action is None:
                slog.log_step_error(step, err or "unknown_error")
                continue

            observation, reward = env.step(action)
            slog.log_step(step=step, action=action, reward=reward, error=None)

            if reward.is_done:
                success = True
                break

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in agent loop")
        slog.log_step_error(step=max(step, 1), error_msg=type(exc).__name__)
    finally:
        env.close()

    final_score = 1.0 if success else 0.0
    slog.log_end(success=success, final_score=final_score)
    logger.info("Agent done task=%s success=%s score=%.1f", task_name, success, final_score)
    return success, slog.step_count, final_score


def run_all_tasks(benchmark: str = BENCHMARK) -> None:
    results = []
    for task_id in range(5):
        success, steps, score = run_agent(task_id=task_id, benchmark=benchmark)
        results.append({"task_id": task_id, "success": success, "steps": steps, "score": score})

    total = sum(r["score"] for r in results)
    sys.stderr.write("\n=== Summary ===\n")
    sys.stderr.write(f"Tasks completed: {sum(1 for r in results if r['success'])}/3\n")
    sys.stderr.write(f"Total score: {total:.2f}/3.00\n")
    sys.stderr.flush()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline agent — Rust Build Fixer")
    parser.add_argument("--task", type=int, choices=[0, 1, 2, 3, 4], default=None,
                        help="0=Easy 1=Medium 2=Hard. Omit to run all.")
    parser.add_argument("--benchmark", default="rust-swe-agent-env")
    args = parser.parse_args()

    if args.task is not None:
        run_agent(task_id=args.task, benchmark=args.benchmark)
    else:
        run_all_tasks(benchmark=args.benchmark)
