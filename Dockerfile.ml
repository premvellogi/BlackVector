FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi==0.116.1 "uvicorn[standard]==0.35.0" pydantic==2.11.7
COPY mock_ml/app.py ./app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8200"]
