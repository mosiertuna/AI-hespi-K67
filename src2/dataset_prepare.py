# dataset_prepare.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def prepare_dataset(csv_path, output_dir='data'):
    df = pd.read_csv(csv_path)
    emotions = df['emotion'].unique()

    # Tạo thư mục train/test cho từng cảm xúc
    for usage in ['Training', 'PublicTest']:
        folder = 'train' if usage == 'Training' else 'test'
        for emotion in emotions:
            os.makedirs(os.path.join(output_dir, folder, str(emotion)), exist_ok=True)

    # Chuyển từng dòng thành ảnh và resize
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Processing images'):
        usage = row['Usage']
        folder = 'train' if usage == 'Training' else ('test' if usage == 'PublicTest' else None)
        if folder is None:
            continue  # Bỏ qua PrivateTest

        pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48)
        resized_img = cv2.resize(pixels, (96, 96))
        emotion = str(row['emotion'])
        img_path = os.path.join(output_dir, folder, emotion, f'{idx}.jpg')
        cv2.imwrite(img_path, resized_img)

if __name__ == '__main__':
    prepare_dataset('fer2013.csv')
