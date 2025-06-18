import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel_h, sobel_v
import os
import glob

class EmotionDetector:
    def __init__(self, model_path=None):
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        self.model = None
        self.face_cascade = None
        
        # Load model
        if model_path:
            self.load_model(model_path)
        else:
            self.auto_load_model()
        
        # Load face detector
        self.load_face_detector()
    
    def auto_load_model(self):
        """Tự động tìm và load model mới nhất"""
        model_files = glob.glob("emotion_svm_model_*.pkl")
        if model_files:
            # Lấy file mới nhất
            latest_model = max(model_files, key=os.path.getctime)
            self.load_model(latest_model)
        else:
            print("❌ Không tìm thấy model file!")
            print("💡 Chạy fer.py trước để train model")
            return False
        return True
    
    def load_model(self, model_path):
        """Load model đã train"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            return False
        return True
    
    def load_face_detector(self):
        """Load Haar Cascade face detector"""
        try:
            # Thử load từ file local
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                # Thử load từ OpenCV built-in
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    raise Exception("Haar Cascade not found")
            print("✅ Face detector loaded")
        except Exception as e:
            print(f"❌ Lỗi load face detector: {e}")
            # Download nếu cần
            self.download_haar_cascade()
    
    def download_haar_cascade(self):
        """Download Haar Cascade file"""
        try:
            import urllib.request
            print("📥 Downloading Haar Cascade...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            print("✅ Haar Cascade downloaded and loaded")
        except Exception as e:
            print(f"❌ Failed to download Haar Cascade: {e}")
    
    def extract_features(self, img):
        """Trích xuất features từ ảnh (giống như training)"""
        # 1. HOG features
        hog_feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, 
                      feature_vector=True, channel_axis=None)
        
        # 2. LBP features
        lbp = local_binary_pattern(img, P=24, R=8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # 3. Statistical features
        stat_features = [
            img.mean(),
            img.std(),
            np.median(img),
            img.min(),
            img.max(),
        ]
        
        # 4. Gradient features
        grad_h = sobel_h(img)
        grad_v = sobel_v(img)
        gradient_features = [
            grad_h.mean(),
            grad_h.std(),
            grad_v.mean(),
            grad_v.std(),
        ]
        
        # Kết hợp tất cả features
        combined_features = np.concatenate([
            hog_feat,
            lbp_hist,
            stat_features,
            gradient_features
        ])
        
        return combined_features
    
    def predict_emotion(self, face_img):
        """Dự đoán cảm xúc từ ảnh khuôn mặt"""
        if self.model is None:
            return "No Model", 0.0
        
        try:
            # Resize về 48x48 và normalize
            face_resized = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_AREA)
            face_normalized = face_resized.astype('float32') / 255.0
            
            # Trích xuất features
            features = self.extract_features(face_normalized)
            
            # Predict
            prediction = self.model.predict(features.reshape(1, -1))
            
            # Lấy probability nếu có
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features.reshape(1, -1))
                confidence = np.max(probabilities)
            else:
                confidence = 1.0  # Nếu không có probability
            
            emotion = self.emotion_labels[prediction[0]]
            return emotion, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0
    
    def run_webcam(self):
        """Chạy detection từ webcam"""
        if self.model is None:
            print("❌ Model chưa được load!")
            return
        
        if self.face_cascade is None:
            print("❌ Face detector chưa được load!")
            return
        
        # Khởi tạo webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Không thể mở webcam!")
            return
        
        print("🎥 Webcam emotion detection started!")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'r' to reset emotion history")
        print("-" * 50)
        
        # Tracking variables
        frame_count = 0
        screenshot_count = 0
        emotion_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không thể đọc từ webcam!")
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(50, 50)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_gray = gray[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence = self.predict_emotion(face_gray)
                
                # Add to history
                if emotion != "Error" and emotion != "No Model":
                    emotion_history.append(emotion)
                    if len(emotion_history) > 10:  # Keep last 10 predictions
                        emotion_history.pop(0)
                
                # Choose color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 0, 255)  # Red - low confidence
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label
                label = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw face ID if multiple faces
                if len(faces) > 1:
                    cv2.putText(frame, f"Face {faces.tolist().index([x,y,w,h])+1}", 
                               (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw info panel
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_count}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Faces: {len(faces)}", (10, info_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Most common emotion in recent history
            if emotion_history:
                most_common = max(set(emotion_history), key=emotion_history.count)
                cv2.putText(frame, f"Trend: {most_common}", (10, info_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Emotion Detection - SVM Model', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"emotion_webcam_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
                screenshot_count += 1
            elif key == ord('r'):
                emotion_history.clear()
                print("🔄 Emotion history reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n📊 Session Summary:")
        print(f"   Total frames: {frame_count}")
        print(f"   Screenshots: {screenshot_count}")
        if emotion_history:
            print(f"   Most detected emotion: {max(set(emotion_history), key=emotion_history.count)}")
        print("✅ Webcam session ended")

def main():
    """Main function"""
    print("🎭 Emotion Detection từ Webcam")
    print("=" * 40)
    
    # Khởi tạo detector
    detector = EmotionDetector()
    
    if detector.model is None:
        print("❌ Không thể load model!")
        print("💡 Hãy chạy fer.py trước để train model")
        return
    
    # Chạy webcam detection
    detector.run_webcam()

if __name__ == "__main__":
    main()