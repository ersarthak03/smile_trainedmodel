FROM python:3.9-slim

# Install system dependencies for dlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download and verify dlib model
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    chmod a+r shape_predictor_68_face_landmarks.dat && \
    [ -s shape_predictor_68_face_landmarks.dat ] || (echo "Model file invalid" && exit 1)

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==20.1.0

# Copy application
COPY . .

# Production server configuration
CMD ["gunicorn", \
    "--bind", "0.0.0.0:$PORT", \
    "--workers", "2", \
    "--threads", "2", \
    "--timeout", "120", \
    "app:app"]
