# File này được sử dụng để test mô hình final_model.h5 được huấn luyện online từ trên Kaggle máy ảo
# Mô hình này được huấn luyện trên tập dữ liệu FER2013
# Mô hình này được sử dụng để nhận diện cảm xúc từ webcam


import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load mô hình đã huấn luyện
model = load_model('final_model.h5')

# Danh sách cảm xúc
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load bộ nhận diện khuôn mặt Haar Cascade
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]

        # Vẽ khung và nhãn
        cv2.rectangle(frame, (x, y-40), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)



    cv2.imshow('Emotion Detection', cv2.resize(frame, (800, 480)))

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
