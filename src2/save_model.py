from tensorflow.keras.models import load_model
import tensorflow as tf

# Định nghĩa lại hàm custom dùng trong mô hình
def gray_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# Load lại model với hàm custom
model = load_model('final_model.h5', custom_objects={'gray_to_rgb': gray_to_rgb})

# Lưu dưới dạng SavedModel (thư mục)
model.export('final_model_savedmodel')
