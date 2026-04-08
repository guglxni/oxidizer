# syntax=docker/dockerfile:1.4
# ^^^ Enables BuildKit features (heredocs, cache mounts). Requires
#     Docker >= 23 or DOCKER_BUILDKIT=1 on older versions.

# Rust Build Fixer RL Environment
# Python 3.10-slim + Rust stable toolchain

FROM python:3.10-slim

LABEL maintainer="rust-swe-agent" \
      version="1.0.0" \
      description="OpenEnv RL Environment for Rust Build Fixing"

# Place CARGO_HOME and RUSTUP_HOME under /app so the non-root appuser
# (who runs the server at runtime) can write the Cargo registry and lock files.
# /usr/local/cargo would be root-owned and deny writes at runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CARGO_HOME=/app/.cargo \
    RUSTUP_HOME=/app/.rustup \
    CARGO_INCREMENTAL=1 \
    PATH="/app/.cargo/bin:$PATH"

# ── System deps + Rust toolchain ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        libssl-dev \
        git \
        ca-certificates \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
       | sh -s -- -y --default-toolchain stable --no-modify-path \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN cargo --version && rustc --version

# ── Pre-warm Cargo registry ───────────────────────────────────────────────────
# `cargo fetch` downloads crate source tarballs for every dependency the tasks
# use, so runtime `cargo check` hits the on-disk cache instead of the network.
# We use printf to avoid requiring heredoc BuildKit support for this layer.
RUN mkdir -p /tmp/cargo_warmup/src \
    && printf '[package]\nname="warmup"\nversion="0.1.0"\nedition="2021"\n\n[dependencies]\nrand="0.8"\nserde={version="1.0",features=["derive"]}\nserde_json="1.0"\nreqwest={version="0.11",features=["blocking","json"]}\ntokio={version="1.0",features=["full"]}\nchrono="0.4"\nregex="1.0"\n' \
       > /tmp/cargo_warmup/Cargo.toml \
    && echo 'fn main(){}' > /tmp/cargo_warmup/src/main.rs \
    && cargo fetch --manifest-path /tmp/cargo_warmup/Cargo.toml \
    && rm -rf /tmp/cargo_warmup

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY env.py inference.py openenv.yaml ./

# ── Non-root user ─────────────────────────────────────────────────────────────
# chown -R covers /app/.cargo and /app/.rustup so appuser can update the
# registry index and Cargo.lock files during runtime cargo check calls.
RUN groupadd -r appgroup \
    && useradd -r -g appgroup appuser \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 7860

# curl is already installed; use it for the health probe (no Python needed).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
