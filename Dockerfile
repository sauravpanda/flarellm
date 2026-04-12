# Multi-stage build for flare-server
#
# Stage 1 — Build
#   Compiles the native server binary with cargo.
# Stage 2 — Runtime
#   Minimal Debian image with only the binary and its shared-library deps.
#
# Usage:
#   docker build -t flare-server .
#   docker run -p 8080:8080 flare-server
#   docker run -p 8080:8080 -v ./models:/models -e MODEL_FILE=/models/model.gguf flare-server

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM rust:1.85-slim AS builder

WORKDIR /build

# Install build-time C deps (linker, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY flare-core       flare-core/
COPY flare-loader     flare-loader/
COPY flare-gpu        flare-gpu/
COPY flare-simd       flare-simd/
COPY flare-server     flare-server/
COPY flarellm         flarellm/
# flare-web is WASM-only; skip it to avoid needing wasm-pack in the builder
COPY flare-web/Cargo.toml flare-web/Cargo.toml
RUN mkdir -p flare-web/src && echo 'fn main() {}' > flare-web/src/lib.rs

RUN cargo build --release -p flarellm-server

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/flare-server /usr/local/bin/flare-server

# Optional: volume mount point for model files
VOLUME ["/models"]

ENV HOST=0.0.0.0
ENV PORT=8080
# Set MODEL_FILE to the path inside /models to load a model on startup, e.g.:
#   -e MODEL_FILE=/models/phi-3-mini.gguf
ENV MODEL_FILE=""

EXPOSE 8080

ENTRYPOINT ["/bin/sh", "-c", \
  "if [ -n \"$MODEL_FILE\" ]; then \
     exec flare-server --model \"$MODEL_FILE\" --host \"$HOST\" --port \"$PORT\"; \
   else \
     exec flare-server --host \"$HOST\" --port \"$PORT\"; \
   fi"]
