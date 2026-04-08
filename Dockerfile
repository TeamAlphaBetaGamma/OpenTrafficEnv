FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -U pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 7860

# HF Spaces typically provides $PORT. Default to 7860 locally.
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
