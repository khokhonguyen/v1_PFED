import cv2
import kagglehub
import pandas as pd 
import numpy as np 

# Tai FER2013
path = kagglehub.dataset_download("msambare/fer2013")
print("Đường dẫn đến các file dữ liệu:", path)

# Đọc dữ liệu
def preprocess_data(data):
    images = []
    labels = []
    for_, row in data.iterrows():
        pixels = np.array(row['pixels'].split(),dtype='float32')
        image = pixels.reshape(48,48)
        # 
        images.append(image)
        labels.append(row['emotion'])
    images =np.array(images)
    labels =np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


        
data = np.read_csv('')