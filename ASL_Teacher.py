import cv2
import time
import argparse
from ultralytics import YOLO
from collections import deque, Counter
import mediapipe as mp
import numpy as np
import json
import os

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser(description="Live ASL hand detection + classification with pose overlay.")
parser.add_argument('--blur', action='store_true', help="Enable Gaussian blur.")
parser.add_argument('--kernel', type=int, help="Kernel size for Gaussian blur (must be odd).")
parser.add_argument('--mode', choices=['free', 'guided'], default='free', help="Choose 'free' for classifier-driven, or 'guided' to practice a fixed letter.")
parser.add_argument('--target_letter', type=str, default=None, help="The ASL letter to practice in guided mode.")
args = parser.parse_args()

# ---------------- Model Load ----------------
detector = YOLO("models/v8m_vanila_3 dtc/weights/best.pt")
classifier = YOLO("models/v8m_blur_7x7 cls/weights/best.pt")

img_size_cls = 224

# ---------------- Bouding Box Scaling ----------------
# Default settings
scale_factor = 1.25
conf_threshold = 0.5
cls_conf_threshold = 0.5

# ---------------- Prediction Buffer ----------------
buffer_size = 10
prediction_buffer = deque(maxlen=buffer_size)

# ---------------- MediaPipe Hands Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ---------------- Constants ----------------
AVERAGE_LANDMARKS_DIR = "./average_landmarks"  # -> change to destination of average landmarks folder in live camera feed
pose_threshold = 0.15  # Threshold for matching a landmark position (normalized units)

HAND_CONNECTIONS_LEFT = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

HAND_CONNECTIONS_RIGHT = [
    (0, 17), (17, 18), (18, 19), (19, 20),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (13, 9), (9, 10), (10, 11), (11, 12),
    (9, 5), (5, 6), (6, 7), (7, 8),
    (5, 1), (1, 2), (2, 3), (3, 4),
    (0, 1)
]

# ---------------- Video Capture ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

cv2.namedWindow("ASL Hand Recognition")
def nothing(x): pass

cv2.namedWindow("Pose Overlay", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Overlay", img_size_cls, img_size_cls)

cv2.namedWindow(f"Hand ({args.kernel}x{args.kernel})", cv2.WINDOW_NORMAL)
cv2.resizeWindow(f"Hand ({args.kernel}x{args.kernel})", img_size_cls, img_size_cls)

# ---------------- Load and Scale Landmarks ----------------
def load_scaled_reference(letter, user_landmarks):
    try:
        with open(os.path.join(AVERAGE_LANDMARKS_DIR, f"{letter.upper()}.json"), 'r') as f:
            avg = json.load(f)
            avg = [(pt[0], pt[1]) for pt in avg]

        user_x = [pt[0] for pt in user_landmarks]
        user_y = [pt[1] for pt in user_landmarks]
        user_width = max(user_x) - min(user_x)
        user_height = max(user_y) - min(user_y)

        avg_x = [pt[0] for pt in avg]
        avg_y = [pt[1] for pt in avg]
        avg_width = max(avg_x) - min(avg_x)
        avg_height = max(avg_y) - min(avg_y)

        scale_x = user_width / avg_width if avg_width else 1
        scale_y = user_height / avg_height if avg_height else 1

        user_origin = user_landmarks[0]
        avg_origin = avg[0]

        scaled = []
        for x, y in avg:
            dx = x - avg_origin[0]
            dy = y - avg_origin[1]
            scaled_x = int(user_origin[0] + dx * scale_x)
            scaled_y = int(user_origin[1] + dy * scale_y)
            scaled.append((scaled_x, scaled_y))

        return scaled
    except Exception as e:
        print(f"Failed to load/scale landmarks for {letter}: {e}")
        return None

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    results = detector.predict(source=frame, conf=conf_threshold, imgsz=320, stream=True)

    hand_crop = None
    crop_coords = None
    for r in results:
        if len(r.boxes) == 0:
            continue

        # Get largest detected hand box
        largest_box = max(r.boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        box_w = x2 - x1
        box_h = y2 - y1
        box_size = int(max(box_w, box_h) * scale_factor)

        half_size = box_size // 2
        x1 = max(x_center - half_size, 0)
        y1 = max(y_center - half_size, 0)
        x2 = min(x_center + half_size, frame.shape[1])
        y2 = min(y_center + half_size, frame.shape[0])

        hand_crop = frame[y1:y2, x1:x2]
        crop_coords = (x1, y1)

        if hand_crop.shape[0] == 0 or hand_crop.shape[1] == 0:
            hand_crop = None
            continue

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        break

    input_img = None
    label_text = ""
    if hand_crop is not None:
        if args.blur:
            if args.kernel % 2 == 0:
                args.kernel += 1
            hand_crop = cv2.GaussianBlur(hand_crop, (args.kernel, args.kernel), 0)

        input_img = cv2.resize(hand_crop, (img_size_cls, img_size_cls))
        results = classifier(input_img, task="classify")
        pred_label = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.data[results[0].probs.top1].item()

        prediction_buffer.append((pred_label, confidence))

        if len(prediction_buffer) == buffer_size:
            labels = [label for label, _ in prediction_buffer]
            most_common = Counter(labels).most_common(1)[0]
            majority_label, count = most_common
            confidences = [conf for label, conf in prediction_buffer if label == majority_label]
            avg_conf = sum(confidences) / len(confidences)

            if avg_conf >= cls_conf_threshold:
                if args.mode == 'guided' and args.target_letter:
                    if majority_label.upper() == args.target_letter.upper():
                        label_text = "Correct!"
                    else:
                        label_text = "Incorrect"
                    predicted_letter = args.target_letter.upper()
                else:
                    label_text = f"{majority_label} ({avg_conf:.2f}, vote {count}/{buffer_size})"
                    predicted_letter = majority_label
            else:
                label_text = "Blank"
                predicted_letter = None
        else:
            predicted_letter = None

        if hand_crop is not None and crop_coords is not None:
            crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            result = hands.process(crop_rgb)

            if result.multi_hand_landmarks:
                h, w, _ = hand_crop.shape
                hand_landmarks = result.multi_hand_landmarks[0]
                user_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                hand_label = result.multi_handedness[0].classification[0].label
                mirrored = (hand_label == "Right")
                connections = HAND_CONNECTIONS_RIGHT if mirrored else HAND_CONNECTIONS_LEFT

                if predicted_letter:
                    ref_landmarks = load_scaled_reference(predicted_letter, user_landmarks)

                    if ref_landmarks and mirrored:
                        origin_x = user_landmarks[0][0]  # x-coordinate of wrist joint
                        ref_landmarks = [(2 * origin_x - x, y) for (x, y) in ref_landmarks]

                    if ref_landmarks:
                        for start_idx, end_idx in connections:
                            if start_idx < len(user_landmarks) and end_idx < len(user_landmarks):
                                ux1, uy1 = user_landmarks[start_idx]
                                ux2, uy2 = user_landmarks[end_idx]
                                rx1, ry1 = ref_landmarks[start_idx]
                                rx2, ry2 = ref_landmarks[end_idx]

                                dist1 = np.linalg.norm(np.array([ux1, uy1]) - np.array([rx1, ry1]))
                                dist2 = np.linalg.norm(np.array([ux2, uy2]) - np.array([rx2, ry2]))
                                color = (0, 255, 0) if dist1 < pose_threshold * w and dist2 < pose_threshold * w else (0, 0, 255)
                                cv2.line(hand_crop, (ux1, uy1), (ux2, uy2), color, 2)

                        for start_idx, end_idx in connections:
                            if start_idx < len(ref_landmarks) and end_idx < len(ref_landmarks):
                                cv2.line(hand_crop, ref_landmarks[start_idx], ref_landmarks[end_idx], (255, 0, 0), 2)

                        for rx, ry in ref_landmarks:
                            cv2.circle(hand_crop, (rx, ry), 3, (255, 0, 0), -1)

                else:
                    # No label yet — draw user landmarks only
                    for start_idx, end_idx in connections:
                        if start_idx < len(user_landmarks) and end_idx < len(user_landmarks):
                            cv2.line(hand_crop, user_landmarks[start_idx], user_landmarks[end_idx], (0, 255, 255), 2)

    if label_text:
        cv2.putText(display_frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("ASL Hand Recognition", display_frame)
    if input_img is not None:
        cv2.imshow(f"Hand ({args.kernel}x{args.kernel})", input_img)
    if hand_crop is not None:
        cv2.imshow("Pose Overlay", hand_crop)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
