FROM python:3.13.7-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TRUSTFABRIC_MODEL_PATH=/app/outputs/models/trustfabric_model.joblib
WORKDIR /app
COPY requirements-production.txt ./
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements-production.txt \
    && useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser trustfabric/ ./trustfabric/
COPY --chown=appuser:appuser outputs/models/trustfabric_model.joblib ./outputs/models/trustfabric_model.joblib
COPY --chown=appuser:appuser outputs/metrics/benchmark.json ./outputs/metrics/benchmark.json
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); assert d.get('status') == 'ok' and d.get('model_loaded') is True"
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
