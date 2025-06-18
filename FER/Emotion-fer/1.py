# Script đơn giản để test nhanh

import cv2
import joblib
import numpy as np
from skimage.feature import hog
import glob
import os

# Load model mới nhất
model_files = glob.glob("emotion_svm_model_*.pkl")
if not model_files:
    print("❌ Không tìm thấy model! Chạy fer.py trước")
    exit()

model_path = max(model_files, key=os.path.getctime)
model = joblib.load(model_path)
print(f"✅ Loaded: {model_path}")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Webcam
cap = cv2.VideoCapture(0)

print("🎥 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract and process face
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Simple HOG features only (faster)
        features = hog(face_normalized, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, feature_vector=True)
        
        try:
            # Predict (might fail if feature dimensions don't match)
            prediction = model.predict(features.reshape(1, -1))
            emotion = emotions[prediction[0]]
            
            # Draw result
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow('Quick Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()