import numpy as np
import cv2
from ultralytics import YOLO
from short import Sort
from datetime import datetime
import pygame
from threading import Thread
import time
import matplotlib.pyplot as plt
import pandas as pd

# Konstanta global
CLASS_NAMES = ['boots', 'helmet', 'no boots', 'no helmet', 'no wearpack', 'wearpack']
MISSING_ITEM_SOUNDS = {
    'no boots': './assets/audio/no boots.mp3',
    'no helmet': './assets/audio/no helmet.mp3',
    'no wearpack': './assets/audio/no wearpack.mp3',
    'no helmet and no boots': './assets/audio/helmet dan boots.mp3',
    'no helmet and no wearpack': './assets/audio/helmet dan wearpack.mp3',
    'no helmet no wearpack no boots': './assets/audio/helmet wearpack boots.mp3',
    'no wearpack and no boots': './assets/audio/no wearpack no boots(2).mp3'
}
THRESHOLD = 0.5
stop_alarm = False
alarm_thread = None

def initialize_audio():
    pygame.mixer.init()

def initialize_video(path):
    return cv2.VideoCapture(path)

def detect_objects(frame, model):
    frame_resized = cv2.resize(frame, (640, 480))
    results = model(frame_resized)[0]
    return results, frame_resized

def load_model(path="./model/model 200.pt"):
    return YOLO(path)

def load_tracker():
    return Sort()

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
                while pygame.mixer.music.get_busy() and time.time() - start_time < 15 and not stop_alarm:
                    time.sleep(0.1)

def combine_missing_items(items):
    combined = set(items)
    if 'no helmet' in combined and 'no wearpack' in combined and 'no boots' in combined:
        combined = {'no helmet no wearpack no boots'}
    elif 'no helmet' in combined and 'no wearpack' in combined:
        combined -= {'no helmet', 'no wearpack'}
        combined.add('no helmet and no wearpack')
    elif 'no helmet' in combined and 'no boots' in combined:
        combined -= {'no helmet', 'no boots'}
        combined.add('no helmet and no boots')
    elif 'no wearpack' in combined and 'no boots' in combined:
        combined -= {'no wearpack', 'no boots'}
        combined.add('no wearpack and no boots')
    return combined

def handle_alarm(detected_items, alarm_triggered):
    global alarm_thread, stop_alarm
    if alarm_triggered and detected_items:
            print(f"Please use PPE: {', '.join(detected_items)}")
            if alarm_thread is None or not alarm_thread.is_alive():
                stop_alarm = False
                alarm_thread = Thread(target=play_alarm, args=(detected_items,))
                alarm_thread.start()
    else:
        stop_alarm = True
        if alarm_thread is not None and alarm_thread.is_alive():
            alarm_thread.join()

def process_detections(frame, results, tracker):
    detected_items = set()
    worn_items = set()
    missing_items = set()
    distances = []

    # for res in results:
    confs = results.boxes.conf.cpu().numpy()
    indices = np.where(confs > THRESHOLD)[0]

    boxes = results.boxes.xyxy.cpu().numpy()[indices].astype(int)
    classes = results.boxes.cls.cpu().numpy()[indices].astype(int)
    confidences = confs[indices]

        # tracks = tracker.update(boxes).astype(int)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        cls = classes[idx]
        label = CLASS_NAMES[cls]
        conf = confidences[idx]

        if label.startswith("no "):
            missing_items.add(label)
        else:
            worn_items.add(label)
            missing_items.discard(f"no {label}")

        distance = estimate_distance([xmin, ymin, xmax, ymax])
        distances.append(distance)

        color = (0, 0, 255) if "no" in label else (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    return combine_missing_items(missing_items), distances, frame, worn_items, missing_items

def draw_fps_and_distance(frame, fps, avg_distance):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    if avg_distance is not None:
        cv2.putText(frame, f"Avg Distance: {avg_distance:.2f} m", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

def main():
    global alarm_thread, stop_alarm
    initialize_audio()
    model = load_model()
    tracker = load_tracker()
    cap = initialize_video("./assets/pengujian single person tidak lengkap-1.mp4")
    prev_time = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    fps_list = []
    detection_results = []

    frame_number = 0

    while cap.isOpened():
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        fps_list.append(fps)

        status, frame = cap.read()
        if not status:
            break
        
        results, frame_resized = detect_objects(frame, model)
        detected_items, distances, frame_annotated, worn_items, missing_items = process_detections(frame_resized, results, tracker)
        # detected_items = combine_missing_items(detected_items)
        handle_alarm(detected_items, bool(detected_items))

        avg_distance = sum(distances) / len(distances) if distances else None
        draw_fps_and_distance(frame_annotated, fps, avg_distance)

        true_alarm = bool(detected_items)
        # false_alarm = not true_alarm

        detection_results.append({
            "Frame": frame_number,
            "Worn Items": ','.join(sorted(worn_items)) if worn_items else 'Tidak Ada',
            "Missing Items": ','.join(sorted(missing_items)) if missing_items else 'Tidak Ada',
            # "Detected Items": ', '.join(detected_items) if detected_items else 'Lengkap',
            "Average Distance": avg_distance if avg_distance else 0,
            "FPS": fps,
            "Alarm Type": "True Alarm" if true_alarm else "False Alarm"
        })
        frame_number += 1

        cv2.imshow("YOLO Detection", frame_annotated)
        out.write(frame_annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Simpan grafik FPS
    plt.figure(figsize=(10, 5))
    plt.plot(fps_list, label='FPS')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.title('FPS over Time')
    plt.legend()
    plt.savefig('fps_graph.png')
    plt.close()
    print("Grafik FPS telah disimpan sebagai 'fps_graph.png'")

    # Simpan hasil deteksi ke CSV
    df = pd.DataFrame(detection_results)
    df.to_csv('detection_results.csv', index=False)
    print("Hasil deteksi telah disimpan sebagai 'detection_results.csv'")

    # Simpan data FPS ke CSV
    fps_df = pd.DataFrame(fps_list, columns=['FPS'])
    fps_df.to_csv('fps_data.csv', index=False)
    print("Data FPS telah disimpan sebagai 'fps_data.csv'")

    # Tampilkan tabel deteksi di terminal
    print(df)

if __name__ == "__main__":
    main()