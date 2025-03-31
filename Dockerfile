FROM python:3.9-slim

# Install system dependencies for dlib (CMake, C++, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment (optional but recommended)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (including .dat file)
COPY . .

# Ensure the .dat file is accessible
RUN chmod +r face_landmark_68_model.dat

# Set the startup command (adjust if needed)
CMD ["python", "app.py"]
