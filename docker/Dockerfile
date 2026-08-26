FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["--config", "configs/smoke_test.yaml"]
