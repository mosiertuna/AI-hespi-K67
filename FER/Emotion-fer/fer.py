import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from skimage.feature import hog
import joblib
import seaborn as sns 
import os
from datetime import datetime

class EmotionTrainer:
    def __init__(self, csv_path='fer2013.csv'):
        self.csv_path = csv_path
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        
    def load_data(self, sample_size=None):
        """Tải và xử lý dữ liệu FER2013"""
        try:
            print("📁 Đang tải dữ liệu FER2013...")
            data = pd.read_csv(self.csv_path)
            print(f"✅ Đã tải {len(data)} mẫu dữ liệu thành công")
            
            if sample_size and sample_size < len(data):
                data = data.sample(n=sample_size, random_state=42)
                print(f"🔄 Sử dụng {sample_size} mẫu để training")
            
            return data
            
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file '{self.csv_path}'")
            return None
    
    def preprocess_images(self, data):
        """Chuyển đổi pixel strings thành arrays và chuẩn hóa"""
        print("🖼️  Đang xử lý ảnh từ pixel strings...")
        
        def pixel_to_array(pixel_string):
            return np.array(pixel_string.split(' '), dtype='float32').reshape(48, 48) / 255.0
        
        X = np.array([pixel_to_array(p) for p in data['pixels']])
        y = data['emotion'].values
        
        print(f"✅ Đã xử lý {len(X)} ảnh kích thước {X[0].shape}")
        return X, y
    
    def extract_advanced_features(self, X):
        """Trích xuất nhiều loại features kết hợp"""
        print("🔍 Đang trích xuất features nâng cao...")
        
        features_list = []
        total = len(X)
        
        for i, img in enumerate(X):
            # 1. HOG features
            hog_feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False, 
                          feature_vector=True, channel_axis=None)
            
            # 2. LBP features (Local Binary Pattern)
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(img, P=24, R=8, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
            
            # 3. Statistical features
            stat_features = [
                img.mean(),
                img.std(),
                np.median(img),
                img.min(),
                img.max(),
            ]
            
            # 4. Gradient features
            from skimage.filters import sobel_h, sobel_v
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
            
            features_list.append(combined_features)
            
            if (i + 1) % 1000 == 0 or i == total - 1:
                print(f"  📊 Đã xử lý {i + 1}/{total} ảnh ({(i+1)/total*100:.1f}%)")
        
        X_features = np.array(features_list)
        print(f"✅ Advanced features shape: {X_features.shape}")
        return X_features
    
    def split_data(self, X_features, y):
        """Chia dữ liệu train/test"""
        print("✂️  Đang chia dữ liệu train/test...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_features, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📊 Train: {len(self.X_train)} mẫu")
        print(f"📊 Test: {len(self.X_test)} mẫu")
        
        # Hiển thị phân bố classes
        unique, counts = np.unique(self.y_train, return_counts=True)
        for emotion_id, count in zip(unique, counts):
            print(f"   {self.emotion_labels[emotion_id]}: {count} mẫu")
    
    def train_improved_svm(self):
        """SVM với hyperparameters tối ưu hơn"""
        print("🚀 Đang huấn luyện Improved SVM...")
        
        # Extensive grid search với các params tối ưu
        param_grid = {
            'C': [1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly'],
            'degree': [2, 3]  # For poly kernel
        }
        
        svm = SVC(random_state=42, probability=True)
        self.model = GridSearchCV(
            svm, param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        start_time = datetime.now()
        print("⏱️ Bắt đầu training...")
        self.model.fit(self.X_train, self.y_train)
        training_time = datetime.now() - start_time
        
        print(f"✅ Training hoàn tất trong {training_time}")
        print(f"🏆 Best parameters: {self.model.best_params_}")
        print(f"🏆 Best CV score: {self.model.best_score_:.4f}")
    
    def evaluate_model(self):
        """Đánh giá model"""
        print("📈 Đang đánh giá model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"🎯 Độ chính xác: {accuracy:.4f}")
        
        # Classification report - SỬA LỖI: chỉ định labels có trong data
        print("\n📋 Báo cáo chi tiết:")
        
        # Lấy các labels thực tế có trong test set
        unique_labels = np.unique(np.concatenate([self.y_test, y_pred]))
        target_names = [self.emotion_labels[i] for i in unique_labels]
        
        print(classification_report(self.y_test, y_pred, 
                              labels=unique_labels, 
                              target_names=target_names, 
                              zero_division=0))
        
        # Hiển thị thông tin về classes
        print(f"\n📊 Classes trong test set: {len(unique_labels)}/7")
        print(f"📊 Missing classes: {set(range(7)) - set(unique_labels)}")
        missing_emotions = [self.emotion_labels[i] for i in set(range(7)) - set(unique_labels)]
        if missing_emotions:
            print(f"📊 Missing emotions: {missing_emotions}")
    
        # Confusion Matrix - cũng cần sửa
        self.plot_confusion_matrix(self.y_test, y_pred, unique_labels)
        
        return accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Vẽ confusion matrix - SỬA LỖI"""
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
    
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    
        # Tạo label names tương ứng
        label_names = [self.emotion_labels[i] for i in labels]
    
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_names,
                   yticklabels=label_names)
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.title(f'Ma trận nhầm lẫn (Confusion Matrix)\n{len(labels)} emotions present')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename=None):
        """Lưu model"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'emotion_svm_model_{timestamp}.pkl'
        
        joblib.dump(self.model, filename)
        print(f"💾 Model đã được lưu: {filename}")
        return filename
    
    def predict_from_image(self, image_path):
        """Dự đoán emotion từ ảnh"""
        if self.model is None:
            print("❌ Model chưa được train!")
            return None
        
        # Load và xử lý ảnh
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("❌ Không thể tải ảnh")
            return None
        
        # Resize về 48x48
        img_resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Trích xuất features giống như training
        hog_feat = hog(img_normalized, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, 
                      feature_vector=True, channel_axis=None)
        
        from skimage.feature import local_binary_pattern
        from skimage.filters import sobel_h, sobel_v
        
        lbp = local_binary_pattern(img_normalized, P=24, R=8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        stat_features = [img_normalized.mean(), img_normalized.std(), 
                        np.median(img_normalized), img_normalized.min(), img_normalized.max()]
        
        grad_h = sobel_h(img_normalized)
        grad_v = sobel_v(img_normalized)
        gradient_features = [grad_h.mean(), grad_h.std(), grad_v.mean(), grad_v.std()]
        
        combined_features = np.concatenate([hog_feat, lbp_hist, stat_features, gradient_features])
        
        # Predict
        prediction = self.model.predict(combined_features.reshape(1, -1))
        emotion = self.emotion_labels[prediction[0]]
        
        return emotion, img_resized

    def load_balanced_data(self, samples_per_emotion=500):
        """Tải dữ liệu cân bằng - số lượng mẫu đều cho mỗi emotion"""
        try:
            print("📁 Đang tải dữ liệu FER2013...")
            data = pd.read_csv(self.csv_path)
            print(f"✅ Đã tải {len(data)} mẫu dữ liệu thành công")
            
            # Hiển thị phân bố ban đầu
            print("\n📊 Phân bố emotions ban đầu:")
            emotion_counts = data['emotion'].value_counts().sort_index()
            for emotion_id, count in emotion_counts.items():
                print(f"   {self.emotion_labels[emotion_id]}: {count} mẫu")
            
            # Lấy đều samples_per_emotion cho mỗi emotion
            balanced_data = []
            print(f"\n🔄 Lấy {samples_per_emotion} mẫu cho mỗi emotion...")
            
            for emotion_id in range(7):  # 0-6 emotions
                emotion_data = data[data['emotion'] == emotion_id]
                
                if len(emotion_data) >= samples_per_emotion:
                    # Lấy random samples_per_emotion mẫu
                    sampled_data = emotion_data.sample(n=samples_per_emotion, random_state=42)
                    balanced_data.append(sampled_data)
                    print(f"   ✅ {self.emotion_labels[emotion_id]}: {samples_per_emotion} mẫu")
                else:
                    # Nếu không đủ, lấy tất cả
                    balanced_data.append(emotion_data)
                    print(f"   ⚠️ {self.emotion_labels[emotion_id]}: chỉ có {len(emotion_data)} mẫu")
            
            # Gộp tất cả lại
            balanced_dataset = pd.concat(balanced_data, ignore_index=True)
            
            # Shuffle data
            balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"\n✅ Dataset cân bằng: {len(balanced_dataset)} mẫu tổng cộng")
            
            # Hiển thị phân bố cuối cùng
            print("\n📊 Phân bố emotions sau khi cân bằng:")
            final_counts = balanced_dataset['emotion'].value_counts().sort_index()
            for emotion_id, count in final_counts.items():
                print(f"   {self.emotion_labels[emotion_id]}: {count} mẫu")
            
            return balanced_dataset
            
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file '{self.csv_path}'")
            print("📥 Tải FER2013 từ: https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset")
            return None

def main():
    """Hàm main - train với dữ liệu cân bằng 300 mẫu/emotion"""
    print("🚀 Training Emotion Detection với Balanced Dataset (300/emotion)")
    print("=" * 60)
    
    trainer = EmotionTrainer('fer2013.csv')
    
    # Load balanced data - 300 mẫu cho mỗi emotion = 2100 mẫu tổng
    data = trainer.load_balanced_data(samples_per_emotion=500)
    if data is None:
        return
    
    # Preprocess
    print("\n🖼️ Step 1: Preprocessing images...")
    X, y = trainer.preprocess_images(data)
    
    # Extract advanced features
    print("\n🔍 Step 2: Extracting advanced features...")
    X_features = trainer.extract_advanced_features(X)
    
    # Split data 
    print("\n✂️ Step 3: Splitting data...")
    trainer.split_data(X_features, y)
    
    # Train Improved SVM
    print("\n🤖 Step 4: Training Improved SVM...")
    trainer.train_improved_svm()
    
    # Evaluate
    print("\n📈 Step 5: Evaluating model...")
    accuracy = trainer.evaluate_model()
    
    # Save model
    model_path = trainer.save_model()
    
    print(f"\n🎉 Training hoàn tất!")
    print(f"🎯 Final Accuracy: {accuracy:.4f}")
    print(f"📁 Model được lưu tại: {model_path}")
    
    # Test với ảnh mẫu nếu có
    test_image = "test_face.jpg"
    if os.path.exists(test_image):
        emotion, img = trainer.predict_from_image(test_image)
        if emotion:
            print(f"🧪 Test prediction: {emotion}")

if __name__ == "__main__":
    main()