PROJECT: EMOTION DETECTION
# AI-hespi-K67


📝  Application: Facial Emotion Detection System 

  🔹 Overview

This project presents a  Facial Emotion Detection System  that can automatically recognize human emotions based on facial expressions using computer vision and deep learning techniques. The application is designed to classify facial emotions into categories such as  Happy ,  Sad ,  Angry ,  Surprised ,  Fearful ,  Disgusted , and  Neutral .

---

  🔹 Objectives

* To build a real-time system that can accurately detect human emotions from facial expressions.
* To apply  Convolutional Neural Networks (CNN)  for image-based emotion classification.
* To demonstrate the potential use cases of emotion recognition in areas like mental health, education, marketing, and entertainment.

---

  🔹 Key Features

* 📷  Real-Time Webcam Detection : Detects faces and emotions live through a webcam interface.
* 📊  Accurate Emotion Classification : Trained on the FER-2013 dataset for 7 emotion classes.
* 🧠  Deep Learning Model : Utilizes CNN architecture for high accuracy in emotion detection.
* 💾  Model Saving & Loading : Includes functionality to save and reuse trained models.
* ⚙️  Preprocessing Pipeline : Converts grayscale images, resizes input, and normalizes pixel values for optimal learning.

---

  🔹 Technologies Used

*  Python 
*  TensorFlow / Keras 
*  OpenCV 
*  NumPy / Pandas 
*  Matplotlib / Seaborn  (for visualization)

---

  🔹 Dataset

*  FER-2013 (Facial Expression Recognition 2013) 

  * Grayscale facial images of size 48x48 pixels
  * Contains \~35,000 images across 7 emotion labels

---

  🔹 Use Cases

*  Mental Health Monitoring  – detect stress or sadness in users
*  Smart Education  – monitor student engagement
*  Human-Computer Interaction  – adapt systems based on user emotions
*  Marketing & Feedback  – understand customer reactions

---

  🔹 Future Improvements

* Use of  transfer learning models  (e.g., MobileNetV2 or EfficientNet) for higher accuracy and speed.
* Expansion to  multi-modal emotion detection  (facial + voice).
* Deployment via  web or mobile application  for real-world testing.

---

*General structure for each folder( src, src1, src2)

Each folder will have:
- 1 folder data after training contains: 2 folder( train + test)
- 1 file dataset_prepare.py to preprocess data from dataset FER2013( you can download from Kaggle)
- 1 model we have after training (model.h5)
- 1 file emotions.py to run model
- 1 file haarcascade_frontalface_default.xml to help us run camera and detection face

LINK FOR DOWNLOAD DATA
- Dataset: https://www.kaggle.com/datasets/deadskull7/fer2013 ( FER2013)
- Data after training:
    + For src: https://drive.google.com/file/d/1Q3YvMMFf_ZSlQCnC202v3yZzucZVQYME/view?usp=sharing
    + For src1: https://drive.google.com/file/d/1klJ9lrbG-_OtWJOp_R3yaOu0fdg7gCBo/view?usp=sharing
    + For src2: https://drive.google.com/file/d/1QQ27iHeeX7gXRj1X4cm6jkuL63HHJdGB/view?usp=sharing

