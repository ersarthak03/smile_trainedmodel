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

# Download and verify model file (direct from dlib)
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    chmod a+r shape_predictor_68_face_landmarks.dat && \
    [ -s shape_predictor_68_face_landmarks.dat ] || (echo "Model file is invalid" && exit 1)

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Verify file exists in final image
RUN ls -lh shape_predictor_68_face_landmarks.dat

CMD ["python", "app.py"]
