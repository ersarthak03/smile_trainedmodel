FROM python:3.9-slim

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Download verified model file
RUN wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    chmod a+r shape_predictor_68_face_landmarks.dat

# 3. Install Python packages (with pinned versions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app
COPY . .

# 5. Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1
EXPOSE $PORT
# 6. Start command (shell form for Railway compatibility)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 --access-logfile - app:app
