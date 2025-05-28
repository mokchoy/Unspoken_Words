import cv2
import time
import argparse
import json
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque, Counter

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser(description="Live ASL hand detection + classification with pose overlay.")
parser.add_argument('--blur', action='store_true', help="Enable Gaussian blur.")
parser.add_argument('--kernel', type=int, help="Kernel size for Gaussian blur (must be odd).")
parser.add_argument('--mode', choices=['free', 'guided'], default='free', help="Choose 'free' or 'guided' mode.")
parser.add_argument('--target_letter', type=str, default=None, help="Letter to practice in guided mode.")
parser.add_argument('--cls_model', type=str, default="models/v8m_blur_7x7 cls/weights/best.pt", help="Path to classification model")
args = parser.parse_args()

# ---------------- Model Load ----------------
#detector = YOLO("models/v8m_vanila_3 dtc/weights/best.pt")
#classifier = YOLO("models/v8m_blur_7x7 cls/weights/best.pt")
detector = YOLO("models/yolov8n-cls.pt")
classifier = YOLO("models/yolov8n-cls.pt")

# ---------------- Constants ----------------
img_size_cls = 224
scale_factor = 1.25
conf_threshold = 0.5
cls_conf_threshold = 0.5
buffer_size = 10
pose_threshold = 0.15
AVERAGE_LANDMARKS_DIR = "./average_landmarks"
prediction_buffer = deque(maxlen=buffer_size)

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

HAND_CONNECTIONS_LEFT = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

HAND_CONNECTIONS_RIGHT = [(0, 17), (17, 18), (18, 19), (19, 20), (0, 13), (13, 14), (14, 15), (15, 16), (13, 9), (9, 10),
                          (10, 11), (11, 12), (9, 5), (5, 6), (6, 7), (7, 8), (5, 1), (1, 2), (2, 3), (3, 4), (0, 1)]

# ---------------- Helper Functions ----------------
def load_scaled_reference(letter, user_landmarks):
    try:
        with open(os.path.join(AVERAGE_LANDMARKS_DIR, f"{letter.upper()}.json"), 'r') as f:
            avg = [(pt[0], pt[1]) for pt in json.load(f)]

        ux, uy = zip(*user_landmarks)
        user_width, user_height = max(ux) - min(ux), max(uy) - min(uy)
        ax, ay = zip(*avg)
        avg_width, avg_height = max(ax) - min(ax), max(ay) - min(ay)
        scale_x, scale_y = user_width / avg_width if avg_width else 1, user_height / avg_height if avg_height else 1

        user_origin, avg_origin = user_landmarks[0], avg[0]
        return [(int(user_origin[0] + (x - avg_origin[0]) * scale_x),
                 int(user_origin[1] + (y - avg_origin[1]) * scale_y)) for x, y in avg]
    except Exception as e:
        print(f"Failed to load/scale landmarks for {letter}: {e}")
        return None

# ---------------- Video Stream ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

cv2.namedWindow("ASL Hand Recognition")
cv2.namedWindow("Pose Overlay", cv2.WINDOW_NORMAL)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    results = detector.predict(source=frame, conf=conf_threshold, imgsz=224, stream=False)

    hand_crop, crop_coords = None, None
    for r in results:
        if r.boxes:
            largest_box = max(r.boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            size = int(max(x2 - x1, y2 - y1) * scale_factor) // 2
            x1, y1, x2, y2 = max(cx - size, 0), max(cy - size, 0), min(cx + size, frame.shape[1]), min(cy + size, frame.shape[0])
            hand_crop = frame[y1:y2, x1:x2]
            crop_coords = (x1, y1)
            if hand_crop.size == 0:
                hand_crop = None
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break

    label_text, predicted_letter = "", None
    if hand_crop is not None:
        if args.blur and args.kernel:
            if args.kernel % 2 == 0: args.kernel += 1
            hand_crop = cv2.GaussianBlur(hand_crop, (args.kernel, args.kernel), 0)

        input_img = cv2.resize(hand_crop, (img_size_cls, img_size_cls))
        results = classifier(input_img, task="classify")
        pred_label = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.data[results[0].probs.top1].item()

        prediction_buffer.append((pred_label, confidence))
        if len(prediction_buffer) == buffer_size:
            majority_label, count = Counter(label for label, _ in prediction_buffer).most_common(1)[0]
            avg_conf = np.mean([conf for label, conf in prediction_buffer if label == majority_label])

            if avg_conf >= cls_conf_threshold:
                if args.mode == 'guided' and args.target_letter:
                    label_text = "Correct!" if majority_label.upper() == args.target_letter.upper() else "Incorrect"
                    predicted_letter = args.target_letter.upper()
                else:
                    label_text = f"{majority_label} ({avg_conf:.2f}, vote {count}/{buffer_size})"
                    predicted_letter = majority_label
            else:
                label_text = "Blank"

        if crop_coords and predicted_letter and confidence > 0.7:
            crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            result = hands.process(crop_rgb)
            if result.multi_hand_landmarks:
                h, w, _ = hand_crop.shape
                landmarks = result.multi_hand_landmarks[0]
                user_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
                hand_label = result.multi_handedness[0].classification[0].label
                mirrored = hand_label == "Right"
                connections = HAND_CONNECTIONS_RIGHT if mirrored else HAND_CONNECTIONS_LEFT

                ref_landmarks = load_scaled_reference(predicted_letter, user_landmarks)
                if ref_landmarks and mirrored:
                    origin_x = user_landmarks[0][0]
                    ref_landmarks = [(2 * origin_x - x, y) for x, y in ref_landmarks]

                if ref_landmarks:
                    for start, end in connections:
                        if start < len(user_landmarks) and end < len(user_landmarks):
                            ux1, uy1 = user_landmarks[start]
                            ux2, uy2 = user_landmarks[end]
                            rx1, ry1 = ref_landmarks[start]
                            rx2, ry2 = ref_landmarks[end]
                            d1 = np.linalg.norm(np.array([ux1, uy1]) - np.array([rx1, ry1]))
                            d2 = np.linalg.norm(np.array([ux2, uy2]) - np.array([rx2, ry2]))
                            color = (0, 255, 0) if d1 < pose_threshold * w and d2 < pose_threshold * w else (0, 0, 255)
                            cv2.line(hand_crop, (ux1, uy1), (ux2, uy2), color, 2)

                    for start, end in connections:
                        if start < len(ref_landmarks) and end < len(ref_landmarks):
                            cv2.line(hand_crop, ref_landmarks[start], ref_landmarks[end], (255, 0, 0), 2)

                    for rx, ry in ref_landmarks:
                        cv2.circle(hand_crop, (rx, ry), 3, (255, 0, 0), -1)

    if label_text:
        cv2.putText(display_frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("ASL Hand Recognition", display_frame)
    if hand_crop is not None:
        cv2.imshow("Pose Overlay", hand_crop)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()