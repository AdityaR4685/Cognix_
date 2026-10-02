# syntax=docker/dockerfile:1
FROM python:3.8-slim

# Saner defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# (Optional) build tools in case wheels need compiling; drop if not needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a named user with UID/GID 1000
RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 -s /bin/bash app

# Ensure the workdir is writable
RUN chown app:app /app


# 1) Leverage layer caching for deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

RUN mkdir /carlaCache
RUN chown -R app:app /carlaCache

# 2) Copy your project code
COPY --chown=app:app config ./config
COPY --chown=app:app carlagen ./carlagen
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app sim.py .

# Drop privileges for runtime
USER app
# Default shell; override with `docker compose run client ...`
CMD ["bash"]

