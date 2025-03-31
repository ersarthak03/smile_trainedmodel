import os
import tempfile
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import dlib

app = Flask(__name__)

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define mouth landmark indices
MOUTH_POINTS = list(range(48, 68))

def calculate_smile_score(landmarks, face_height):
    """Calculate normalized smile score (0-100)"""
    left_corner = np.array([landmarks.part(48).x, landmarks.part(48).y])
    right_corner = np.array([landmarks.part(54).x, landmarks.part(54).y])
    upper_lip = np.array([landmarks.part(51).x, landmarks.part(51).y])
    lower_lip = np.array([landmarks.part(57).x, landmarks.part(57).y])

    mouth_width = np.linalg.norm(left_corner - right_corner)
    mouth_height = np.linalg.norm(upper_lip - lower_lip)
    mouth_openness = (mouth_height / face_height) * 100
    lip_curvature = np.linalg.norm(
        np.array([landmarks.part(50).x, landmarks.part(50).y]) -
        np.array([landmarks.part(58).x, landmarks.part(58).y])
    ) / face_height * 80

    smile_score = min(100, max(0, mouth_openness + lip_curvature))
    
    # Rescale for better sensitivity
    if smile_score > 10:
        smile_score = ((smile_score - 10) / 40) * 90 + 10
    
    return round(min(100, smile_score), 2)

def draw_annotations(image, face, landmarks, score):
    """Draw face bounding box, landmarks, and score"""
    color = (0, 255, 0) if score > 50 else (0, 0, 255)
    
    # Draw bounding box
    cv2.rectangle(image, (face.left(), face.top()), 
                 (face.right(), face.bottom()), color, 2)
    
    # Draw landmarks
    for i in MOUTH_POINTS:
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    
    # Draw score
    cv2.putText(image, f"Smile: {score}%", 
               (face.left(), face.top() - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    """API endpoint for smile detection"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Read image directly from memory
        file = request.files['image']
        img_bytes = file.read()
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if not faces:
            return jsonify({"error": "No faces detected"}), 400

        results = []
        for face in faces:
            landmarks = predictor(gray, face)
            face_height = face.bottom() - face.top()
            score = calculate_smile_score(landmarks, face_height)
            draw_annotations(image, face, landmarks, score)
            results.append({"score": score, "box": [
                face.left(), face.top(), 
                face.right(), face.bottom()
            ]})

        # Save processed image to temp file
        _, temp_path = tempfile.mkstemp(suffix='.jpg')
        cv2.imwrite(temp_path, image)

        return send_file(
            temp_path,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='processed.jpg',
            add_etags=False
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp files if they exist
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
