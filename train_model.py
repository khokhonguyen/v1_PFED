from model import build_model
from data_processing import load_data

# Đọc dữ liệu
data_dir = "datasets/fer2013"
train_generator, validation_generator = load_data(data_dir)

# Xây dựng mô hình
model = build_model()

# Huấn luyện mô hình
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
# Huấn luyện trên dữ liệu gốc và dữ liệu bổ sung
mixed_train_data = augment_mixed_emotions(train_generator)
history = model.fit(mixed_train_data, epochs=10, validation_data=validation_generator)

# Lưu mô hình đã huấn luyện
model.save("emotion_model.h5")
