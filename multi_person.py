# Let's merge both code bases: the main code with PPE detection and the additional code for person tracking using a separate YOLO model.
# We'll assume that the person-detection model is `yolov8n.pt` and the PPE model is the user's custom model.

# Final revised version of your program
# ID person tracking dihapus dan alarm multi-person disederhanakan

import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import pygame
from threading import Thread
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

PPE_CLASS_NAMES = ['boots', 'helmet', 'no boots', 'no helmet', 'no wearpack', 'wearpack']
MISSING_ITEM_SOUNDS = {
    'no boots': './assets/audio/no boots.mp3',
    'no helmet': './assets/no helmet.mp3',
    'no wearpack': './assets/no wearpack.mp3',
    'no helmet and no boots': './assets/helmet dan boots.mp3',
    'no helmet and no wearpack': './assets/helmet dan wearpack.mp3',
    'no helmet no wearpack no boots': './assets/helmet wearpack boots.mp3',
    'no wearpack and no boots': './assets/no wearpack no boots(2).mp3'
}
THRESHOLD = 0.5
stop_alarm = False
alarm_thread = None

def initialize_audio():
    pygame.mixer.init()

def load_model(path):
    return YOLO(path)

def estimate_distance(box):
    width = box[2] - box[0]
    reference_width = 100
    reference_distance = 2.0
    return reference_distance * (reference_width / max(width, 1))

def play_alarm(missing_items):
    global stop_alarm
    start_time = time.time()
    while time.time() - start_time < 15 and not stop_alarm:
        for item in missing_items:
            if stop_alarm:
                break
            if item in MISSING_ITEM_SOUNDS:
                pygame.mixer.music.load(MISSING_ITEM_SOUNDS[item])
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not stop_alarm:
                    time.sleep(0.1)

def play_general_alarm():
    pygame.mixer.music.load('./assets/audio/peringatan multi person.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() and not stop_alarm:
        time.sleep(0.1)

def combine_missing_items(items):
    combined = set(items)
    if {'no helmet', 'no wearpack', 'no boots'}.issubset(combined):
        return {'no helmet no wearpack no boots'}
    elif {'no helmet', 'no wearpack'}.issubset(combined):
        return {'no helmet and no wearpack'}
    elif {'no helmet', 'no boots'}.issubset(combined):
        return {'no helmet and no boots'}
    elif {'no wearpack', 'no boots'}.issubset(combined):
        return {'no wearpack and no boots'}
    return combined

def detect_objects(frame, model):
    resized = cv2.resize(frame, (640, 480))
    results = model(resized)[0]
    return results, resized

def process_detections(frame, results):
    worn_items = set()
    missing_items = []
    distances = []
    confs = results.boxes.conf.cpu().numpy()
    indices = np.where(confs > THRESHOLD)[0]
    boxes = results.boxes.xyxy.cpu().numpy()[indices].astype(int)
    classes = results.boxes.cls.cpu().numpy()[indices].astype(int)
    for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        label = PPE_CLASS_NAMES[classes[idx]]
        if label.startswith("no "):
            missing_items.append(label)
        else:
            worn_items.add(label)
        distance = estimate_distance([xmin, ymin, xmax, ymax])
        distances.append(distance)
        color = (0, 0, 255) if "no" in label else (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    return combine_missing_items(missing_items), distances, frame, worn_items, missing_items

def aggregate_missing_items_count(missing_items_list):
    counts = defaultdict(int)
    for missing in missing_items_list:
        for item in missing:
            counts[item] += 1
    return dict(counts)

def handle_alarm(detected_items, is_multi_person):
    global alarm_thread, stop_alarm
    if is_multi_person:
        print("⚠️ PERINGATAN: Beberapa orang tidak memakai APD lengkap")
        if alarm_thread is None or not alarm_thread.is_alive():
            stop_alarm = False
            alarm_thread = Thread(target=play_general_alarm)
            alarm_thread.start()
    elif detected_items:
        print("Peringatan: Penggunaan APD tidak lengkap")
        if alarm_thread is None or not alarm_thread.is_alive():
            stop_alarm = False
            alarm_thread = Thread(target=play_alarm, args=(detected_items,))
            alarm_thread.start()
    else:
        stop_alarm = True
        if alarm_thread and alarm_thread.is_alive():
            alarm_thread.join()

def draw_fps_and_distance(frame, fps, avg_distance):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    if avg_distance is not None:
        cv2.putText(frame, f"Avg Distance: {avg_distance:.2f} m", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

def main():
    global alarm_thread, stop_alarm
    initialize_audio()
    model_ppe = load_model("./model/model 200.pt")
    cap = cv2.VideoCapture("./assets/pengujian multi person tidak lengkap semua-1.mp4")
    prev_time = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(f"output_{timestamp}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

    detection_results = []
    fps_list = []
    frame_number = 0

    while cap.isOpened():
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        fps_list.append(fps)

        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results, frame_resized = detect_objects(frame, model_ppe)
        detected_items, distances, frame_annotated, worn_items, missing_items = process_detections(frame_resized, results)

        all_missing_summary = aggregate_missing_items_count([missing_items])
        true_alarm = bool(detected_items)

        handle_alarm(detected_items, true_alarm)

        avg_distance = np.mean(distances) if distances else None
        draw_fps_and_distance(frame_annotated, fps, avg_distance)

        detection_results.append({
            "Frame": frame_number,
            "Worn Items": ','.join(sorted(worn_items)) if worn_items else 'Tidak Ada',
            "Missing Items": ','.join(sorted(missing_items)) if missing_items else 'Tidak Ada',
            "Missing Summary": str(all_missing_summary),
            "Avg Distance": avg_distance if avg_distance else 0,
            "FPS": fps,
            "Alarm Type": "True Alarm" if true_alarm else "False_Alarm"
        })

        frame_number += 1
        out.write(frame_annotated)
        cv2.imshow("PPE Detection", frame_annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    plt.figure(figsize=(10, 5))
    plt.plot(fps_list, label='FPS')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.title('FPS over Time')
    plt.legend()
    plt.savefig('fps_graph.png')
    plt.close()

    df = pd.DataFrame(detection_results)
    df.to_csv('detection_results.csv', index=False)
    fps_df = pd.DataFrame(fps_list, columns=['FPS'])
    fps_df.to_csv('fps_data.csv', index=False)
    print("Detection and FPS data saved.")

if __name__ == "__main__":
    main()