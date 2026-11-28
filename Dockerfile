# ProofPath application image — shared by the API and the background worker.
#
# Sub-part 2.2.1a: the `worker` service in docker-compose.yml builds this image
# and overrides the command. The same image will serve the FastAPI app in a later
# slice, so the codebase is installed once and shared across services.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the proofpath package and its pinned runtime dependencies. Every core
# dependency ships a manylinux / cp311 wheel, so no build toolchain is required
# and the image stays small.
COPY pyproject.toml alembic.ini ./
COPY src/ ./src/
COPY migrations/ ./migrations/

ARG ENABLE_GUARDRAILS=false
RUN case "$(echo "$ENABLE_GUARDRAILS" | tr '[:upper:]' '[:lower:]')" in \
      1|true|yes|on|t|y) pip install '.[guardrails]' ;; \
      *) pip install . ;; \
    esac

# Run the worker as a non-root user.
RUN useradd --create-home --uid 1000 proofpath \
    && chown -R proofpath:proofpath /app
USER proofpath

# Default command runs the background worker; override per service in compose.
CMD ["python", "-m", "proofpath.workers"]
