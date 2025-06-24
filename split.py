import cv2
import os

# Path ke video
video_path = './assets/output_20250608_224000.mp4'
output_dir = './frames_output_mp_ssapd/'
os.makedirs(output_dir, exist_ok=True)

# Buka video
cap = cv2.VideoCapture(video_path)
frame_interval = 95  # setiap 95 frame
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Total frame disimpan: {saved_count}")
