FROM python:3.11-slim

WORKDIR /app

# copy only requirements first to leverage layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of backend code
COPY . .

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# run using gunicorn for production-like behavior
CMD ["gunicorn", "server:app", "-b", "0.0.0.0:5000", "--workers", "2", "--threads", "4"]
