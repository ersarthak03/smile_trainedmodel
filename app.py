import os
import dlib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # For handling CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize dlib models
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

@app.route('/')
def home():
    """Root endpoint to verify service is running"""
    return "Face Detection Service is Running"

@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    """API endpoint for smile detection"""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image provided"}), 400
    
    try:
        # Read image
        file = request.files['image']
        img = dlib.load_rgb_image(file)
        
        # Detect faces
        faces = detector(img)
        if not faces:
            return jsonify({
                "status": "error",
                "message": "No faces detected",
                "hint": "Try a clearer front-facing photo"
            }), 400
        
        # Process first face found
        face = faces[0]
        landmarks = predictor(img, face)
        
        # Calculate smile percentage
        mouth_width = landmarks.part(54).x - landmarks.part(48).x
        mouth_open = landmarks.part(66).y - landmarks.part(62).y
        smile_percent = min(100, max(0, (mouth_open / mouth_width) * 300))
        
        return jsonify({
            "status": "success",
            "smile_percentage": round(smile_percent, 2),
            "face_location": {
                "left": face.left(),
                "top": face.top(),
                "right": face.right(),
                "bottom": face.bottom()
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "hint": "Ensure you're sending a valid image file (JPEG/PNG)"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
