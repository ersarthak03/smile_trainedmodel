import os
import dlib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Read image
        file = request.files['image']
        img = dlib.load_rgb_image(file)
        
        # Detect faces
        faces = detector(img)
        if not faces:
            return jsonify({"error": "No faces detected"}), 400
        
        # Get first face
        face = faces[0]
        landmarks = predictor(img, face)
        
        # Calculate smile percentage (simplified)
        mouth_width = landmarks.part(54).x - landmarks.part(48).x
        mouth_open = landmarks.part(66).y - landmarks.part(62).y
        smile_percent = min(100, max(0, (mouth_open / mouth_width) * 300))
        
        return jsonify({
            "smile_percentage": round(smile_percent, 2),
            "face_location": {
                "left": face.left(),
                "top": face.top(),
                "right": face.right(),
                "bottom": face.bottom()
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
