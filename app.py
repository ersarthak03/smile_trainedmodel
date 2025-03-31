import os
import dlib
from flask import Flask

app = Flask(__name__)

MODEL_FILE = "shape_predictor_68_face_landmarks.dat"

def verify_model():
    """Verify the model file exists and is valid"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file missing at {os.path.abspath(MODEL_FILE)}")
    
    file_size = os.path.getsize(MODEL_FILE)
    if file_size < 95_000_000:  # Expected ~96MB
        raise ValueError(f"Model file too small ({file_size} bytes)")

try:
    verify_model()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_FILE)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"\n❌ Model loading failed: {str(e)}")
    print("Debugging info:")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files present: {os.listdir()}")
    print(f"Model file size: {os.path.getsize(MODEL_FILE) if os.path.exists(MODEL_FILE) else 'MISSING'}")
    raise

@app.route('/')
def home():
    return "Face Detection Service is Running"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
