import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from collections import Counter
import os

# Load mô hình đã huấn luyện
model = load_model("emotion_model.h5")

# Cảm xúc tương ứng với các nhãn
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Ánh xạ cảm xúc sang trạng thái tâm lý học
emotion_to_psychology = {
    "Happy": "Positive reinforcement",
    "Sad": "Possible mild depression",
    "Angry": "Frustration or high stress",
    "Fear": "Anxiety or fearfulness",
    "Surprise": "Heightened attention",
    "Neutral": "Balanced state",
}

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Lấy kích thước màn hình
monitor = get_monitors()[0]  # Lấy màn hình chính
screen_width = monitor.width
screen_height = monitor.height

# Cài đặt kích thước cửa sổ video để full width màn hình
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)  # Chiều rộng màn hình
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)  # Chiều cao màn hình

# Khởi tạo các biến theo dõi trạng thái
emotion_data = []  # Lưu trữ dữ liệu cảm xúc
start_time = time.time()
emotion_changes = 0  # Số lần cảm xúc thay đổi đột ngột
emotion_change_times = []  # Lưu trữ thời gian thay đổi cảm xúc đột ngột
emotion_change_details = []  # Lưu chi tiết về sự thay đổi cảm xúc (từ -> đến)
emotion_change_count = 0  # Đếm số lần chuyển đổi cảm xúc

# Khởi tạo hàm phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Không thể tải mô hình phát hiện khuôn mặt.")
    exit()


# Hàm để khoanh vùng và tracking khuôn mặt
def track_face(frame):
    global tracking_face
    if tracking_face is not None:
        (x, y, w, h) = tracking_face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ bounding box
        return frame, (x, y, w, h)
    return frame, None


# Hàm chụp ảnh khi có thay đổi cảm xúc đột ngột
def capture_screenshot(frame, elapsed_time, current_emotion, filename="screenshot.png"):
    screenshot_path = os.path.join("screenshots", filename)
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    # Lọc và lấy 4 cảm xúc mạnh nhất
    top_emotions = sorted(
        zip(emotion_labels, probabilities), key=lambda x: x[1], reverse=True
    )[:4]

    # Tạo chuỗi hiển thị cảm xúc với xác suất
    emotion_text = " | ".join(
        [f"{emotion}: {prob*100:.1f}%" for emotion, prob in top_emotions]
    )

    # Chú thích cảm xúc dưới thời gian, vẫn ở góc trái trên ảnh
    cv2.putText(
        frame,
        f"Emotion: {current_emotion} | {emotion_text}",
        (20, 100),  # Vị trí dưới thời gian
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Lưu ảnh và kiểm tra xem ảnh có được lưu thành công không
    result = cv2.imwrite(screenshot_path, frame)
    if result:
        print(f"Screenshot saved at: {screenshot_path}")
    else:
        print(f"Failed to save screenshot at: {screenshot_path}")

    # Lưu ảnh và kiểm tra xem ảnh có được lưu thành công không
    result = cv2.imwrite(screenshot_path, frame)
    if result:
        print(f"Screenshot saved at: {screenshot_path}")
    else:
        print(f"Failed to save screenshot at: {screenshot_path}")


# Hàm phân tích sự thay đổi cảm xúc
def analyze_emotion_change(last_emotion, current_emotion):
    # Nếu cảm xúc thay đổi quá nhanh hoặc đối lập (dựa trên các cảm xúc đối nghịch)
    opposite_emotions = {
        "Angry": ["Happy", "Sad", "Neutral"],
        "Fear": ["Happy", "Sad", "Neutral"],
        "Happy": ["Angry", "Fear", "Sad"],
        "Sad": ["Happy", "Fear", "Neutral"],
        "Surprise": ["Neutral", "Happy"],
        "Neutral": ["Happy", "Fear", "Angry", "Sad"],
    }
    if current_emotion in opposite_emotions.get(last_emotion, []):
        return True  # Có thay đổi bất ngờ
    return False


while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong mỗi frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        # Chọn khuôn mặt có diện tích lớn nhất
        faces = sorted(
            faces, key=lambda x: x[2] * x[3], reverse=True
        )  # Sắp xếp theo diện tích (w * h)
        x, y, w, h = faces[0]  # Chọn khuôn mặt có diện tích lớn nhất
        tracking_face = (x, y, w, h)

        # Cắt khuôn mặt từ ảnh và chuẩn bị cho mô hình dự đoán
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (48, 48))  # Đảm bảo kích thước phù hợp với mô hình
        face = face / 255.0  # Chuẩn hóa dữ liệu
        face = np.reshape(face, (1, 48, 48, 1))  # Định dạng lại cho mô hình

        # Dự đoán cảm xúc
        prediction = model.predict(face)
        probabilities = prediction[0]

        # Lấy cảm xúc với xác suất > 20%
        significant_indices = np.where(probabilities > 0.2)[0]  # Ngưỡng tùy chọn
        significant_emotions = [emotion_labels[i] for i in significant_indices]
        significant_probs = [probabilities[i] for i in significant_indices]

        # Xử lý tối đa 3 cảm xúc mạnh nhất
        top_emotions = sorted(
            zip(significant_emotions, significant_probs),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        # Lưu trữ dữ liệu cảm xúc vào danh sách
        current_emotion = top_emotions[0][0] if top_emotions else "Unknown"
        emotion_data.append((current_emotion, time.time()))

        # Kiểm tra thời gian từ lúc bắt đầu chương trình
        elapsed_time = time.time() - start_time

        # Hiển thị thời gian trên ảnh
        cv2.putText(
            frame,
            f"Time: {elapsed_time:.2f} sec",
            (20, 50),  # Vị trí góc trái trên
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Kiểm tra sự thay đổi cảm xúc đột ngột
        if len(emotion_data) > 1:
            last_emotion, last_time = emotion_data[-2]
            current_emotion, current_time = emotion_data[-1]
            if analyze_emotion_change(last_emotion, current_emotion):
                print(
                    f"Sudden emotion change detected: {last_emotion} -> {current_emotion}"
                )

                # Lưu thời gian thay đổi cảm xúc và chi tiết thay đổi
                emotion_change_times.append(elapsed_time)
                emotion_change_details.append(
                    f"Emotion changed from {last_emotion} to {current_emotion} at: {elapsed_time:.2f} sec"
                )

                # Tăng số lần chuyển đổi cảm xúc
                emotion_change_count += 1

                # Chụp ảnh với tên file theo thời gian và cảm xúc
                capture_screenshot(
                    frame,
                    elapsed_time,
                    current_emotion,
                    filename=f"{elapsed_time:.2f}_{current_emotion}.png",
                )

    # Hiển thị ảnh với dự đoán và bounding box
    frame, _ = track_face(frame)

    # Hiển thị ảnh với dự đoán
    cv2.imshow("Emotion Recognition", frame)

    # Dừng lại khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tính toán tỷ lệ cảm xúc và hiển thị bảng thống kê khi kết thúc
emotion_counts = Counter([emotion[0] for emotion in emotion_data])
total_emotions = sum(emotion_counts.values())

# Tính toán phần trăm các cảm xúc
emotion_percentages = {
    emotion: (count / total_emotions) * 100 for emotion, count in emotion_counts.items()
}

# Hiển thị bảng thống kê
print("Emotion Stats over the entire session:")
for emotion, percentage in emotion_percentages.items():
    print(f"{emotion}: {percentage:.2f}%")

# Cảm xúc chiếm phần lớn và ít nhất
max_emotion = max(emotion_percentages, key=emotion_percentages.get)
min_emotion = min(emotion_percentages, key=emotion_percentages.get)

# In kết quả phân tích tâm lý học
print(
    f"Emotion with highest percentage: {max_emotion} ({emotion_percentages[max_emotion]:.2f}%)"
)
print(
    f"Emotion with lowest percentage: {min_emotion} ({emotion_percentages[min_emotion]:.2f}%)"
)

# In số lần chuyển đổi cảm xúc
print(f"Total number of emotion changes: {emotion_change_count}")

# Dựa trên phân tích:
if max_emotion == "Fear":
    print("Analysis: The person is likely to be experiencing anxiety or fear.")
elif max_emotion == "Happy":
    print("Analysis: The person is likely to be in a positive, happy mood.")
elif max_emotion == "Sad":
    print("Analysis: The person may be experiencing mild depression.")
else:
    print("Analysis: The person is in a neutral emotional state.")

# Hiển thị các thời gian thay đổi cảm xúc đột ngột với chi tiết
print("Emotion change details:")
for change_detail in emotion_change_details:
    print(change_detail)

# Hiển thị biểu đồ tròn phân phối cảm xúc
plt.pie(
    emotion_percentages.values(),
    labels=emotion_percentages.keys(),
    autopct="%1.1f%%",
    startangle=90,
)
plt.title("Emotion Distribution")
plt.show()

# Đóng camera và cửa sổ
cap.release()
cv2.destroyAllWindows()
