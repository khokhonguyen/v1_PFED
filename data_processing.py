import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Tạo nhãn cảm xúc hỗn hợp
def augment_mixed_emotions(data_generator):
    """
    Tăng cường dữ liệu với nhãn cảm xúc hỗn hợp (Mixed Emotions).
    """
    augmented_data = []
    for batch_x, batch_y in data_generator:
        for i in range(len(batch_x)):
            if batch_y[i].sum() > 1:  # Nếu nhãn cảm xúc hỗn hợp
                augmented_data.append((batch_x[i], batch_y[i]))
        if len(augmented_data) >= len(data_generator):  # Giới hạn số lượng
            break
    return np.array(augmented_data)


def load_data(data_dir, target_size=(48, 48)):
    # Đọc dữ liệu từ thư mục
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=target_size,
        batch_size=32,
        class_mode="sparse",
        color_mode="grayscale",
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(data_dir, "validation"),
        target_size=target_size,
        batch_size=32,
        class_mode="sparse",
        color_mode="grayscale",
    )

    # Tăng cường dữ liệu hỗn hợp
    train_generator_augmented = augment_mixed_emotions(train_generator)

    return train_generator_augmented, validation_generator
