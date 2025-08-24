FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate data, build features, train model at build time (small sample to keep image light)
RUN python -m src.data.make_dataset --n-samples 5000 --fraud-rate 0.02 && \    python -m src.features.build_features && \    python -m src.models.train

EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]