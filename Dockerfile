FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 INFERENCE_BACKEND=onnx REQUIRE_MODEL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PORT=8000
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY app ./app
COPY ml/__init__.py ml/labels.py ml/onnx_inference.py ./ml/
COPY models/backbone.onnx models/heads.npz models/runtime.json models/policy.json models/metadata.json models/metrics.json models/policy_metrics.json ./models/
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --limit-concurrency 4"]
