FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "scripts/run_api.py"]
