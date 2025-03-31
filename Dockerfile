FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download the official dlib model file (always fresh)
RUN wget -O shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    mv shape_predictor_68_face_landmarks.dat face_landmark_68_model.dat && \
    chmod a+r face_landmark_68_model.dat

# Verify model file integrity
RUN ls -lh face_landmark_68_model.dat && \
    [ -s face_landmark_68_model.dat ] || (echo "Model file is invalid" && exit 1)

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import os; assert os.path.exists('face_landmark_68_model.dat')"

CMD ["python", "app.py"]
