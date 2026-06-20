FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY model.pkl .
COPY sample_input.json .

ENTRYPOINT ["python", "src/inference.py"]
CMD ["sample_input.json"]