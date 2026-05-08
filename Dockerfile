# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# PI-GINOT Agentic Studio — External Deployment Container
#
# Build:  docker build -t pi-ginot-studio .
# Run:    docker run -p 8501:8501 --env-file .env pi-ginot-studio
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (two-stage for caching)
COPY requirements-external.txt ./
RUN pip install --no-cache-dir -r requirements-external.txt

# Copy project source
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "llm_agents/Home.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
