from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import dlib
import os

app = Flask(__name__)
application=app

# Load Dlib's face detector and landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmark_68_model.dat")

# Define mouth landmark indices
MOUTH = list(range(48, 68))

def calculate_smile_score(landmarks, face_height):
    """
    Improved smile scoring with normalization.
    """
    left_corner = np.array([landmarks.part(48).x, landmarks.part(48).y])
    right_corner = np.array([landmarks.part(54).x, landmarks.part(54).y])
    upper_lip = np.array([landmarks.part(51).x, landmarks.part(51).y])
    lower_lip = np.array([landmarks.part(57).x, landmarks.part(57).y])

    # Compute mouth width and height
    mouth_width = np.linalg.norm(left_corner - right_corner)
    mouth_height = np.linalg.norm(upper_lip - lower_lip)

    # Compute lip curvature (distance between mid-upper and mid-lower lips)
    lip_curvature = np.linalg.norm(np.array([landmarks.part(50).x, landmarks.part(50).y]) -
                                   np.array([landmarks.part(58).x, landmarks.part(58).y]))

    # Normalize values relative to face height
    mouth_openness = (mouth_height / face_height) * 100  
    curvature_effect = (lip_curvature / face_height) * 80  

    # Compute initial smile score
    smile_score = mouth_openness + curvature_effect

    # Ensure score is within 0-100
    smile_score = max(0, min(smile_score, 100))

    # Adjust scaling: If >10%, map 10-50 to 10-100%
    if smile_score > 10:
        smile_score = ((smile_score - 10) / (50 - 10)) * 90 + 10
        smile_score = min(100, smile_score)  # Cap at 100

    return round(smile_score, 2)

def draw_landmarks(image, landmarks, color=(255, 0, 0)):
    """ Draw facial landmarks as dots. """
    for i in MOUTH:
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(image, (x, y), 2, color, -1)  # Blue dots

def draw_mouth_curve(image, landmarks, color=(0, 255, 255)):
    """ Draw a curve around the lips. """
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in MOUTH], np.int32)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_path = "uploaded_image.jpg"
    file.save(image_path)

    # Load and process image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    smile_scores = []

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Compute face height for normalization
        face_height = face.bottom() - face.top()

        # Calculate smile score using improved formula
        smile_score = calculate_smile_score(landmarks, face_height)
        smile_scores.append(smile_score)

        # Draw landmarks and mouth curve
        draw_landmarks(image, landmarks, color=(255, 0, 0))  # Blue dots
        draw_mouth_curve(image, landmarks, color=(0, 255, 255))  # Yellow curve

        # Draw bounding box
        color = (0, 255, 0) if smile_score > 50 else (0, 0, 255)
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        cv2.putText(image, f"Smile: {smile_score}%", (face.left(), face.top() - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save processed image
    processed_image_path = "output.jpg"
    cv2.imwrite(processed_image_path, image)

    # JSON response
    response = {
        "message": "Smile detection completed!",
        "smile_scores": smile_scores
    }

    return send_file(processed_image_path, mimetype='image/jpeg'), 200, response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
