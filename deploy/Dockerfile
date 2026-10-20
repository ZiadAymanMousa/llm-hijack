# Phase 1 image: vLLM (serves the models) + this project (runs the experiments).
FROM vllm/vllm-openai:v0.30.0

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts
COPY deploy ./deploy
# Editable install: the package finds configs/ and prompts/ next to src/ in /app.
RUN pip install --no-cache-dir -e . \
    && python3 -c "import hijack.run, hijack.replay, hijack.analyze; from hijack.views import STEPGUARD_TEMPLATE"

# Results (and the Hugging Face cache) live on the mounted Azure Files share.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/results/hf-cache

ENTRYPOINT ["bash", "deploy/run_phase1.sh"]
