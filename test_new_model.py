import cv2
import mediapipe as mp
import serial
import time
import math
from collections import deque

# ─────────────────────────────────────────────
# Serial and Tracker Initialization
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

cap = cv2.VideoCapture(0)

trackers = []
tracker_boxes = []
tracker_labels = []
tracker_initialized = False
tracker_history = deque(maxlen=5)

last_command = None
last_switch_time = 0
switch_cooldown = 2
switch_candidate_idx = None
switch_hold_frames = 0
required_hold_frames = 15
target_idx = None

# ─────────────────────────────────────────────
# Load DNN for person detection
model_dir = "mobilenet-ssd/"
net = cv2.dnn.readNetFromCaffe(
    model_dir + "MobileNetSSD_deploy.prototxt",
    model_dir + "MobileNetSSD_deploy.caffemodel"
)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ─────────────────────────────────────────────
# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=3)
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# Helper Functions
def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return dist < 0.05

def count_raised_fingers(hand_landmarks):
    def is_up(tip, pip):
        return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y

    index_up = is_up(8, 6)
    middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y

    if index_up and middle_down and ring_down and pinky_down:
        return 1
    return 0

def count_total_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    count = 0
    for tip in tip_ids[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# ─────────────────────────────────────────────
# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    (h, w) = frame.shape[:2]
    frame_center_x = w // 2
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    person_boxes = []
    for i, tracker in enumerate(trackers):
        success, box = tracker.update(frame)
        if success:
            (x, y, wb, hb) = [int(v) for v in box]
            person_boxes.append((x, y, x + wb, y + hb))
            cv2.rectangle(frame, (x, y), (x + wb, y + hb), (0, 255, 255), 2)
            cv2.putText(frame, f"Person {tracker_labels[i]}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if not tracker_initialized:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and CLASSES[int(detections[0, 0, i, 1])] == "person":
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                bbox = (startX, startY, endX - startX, endY - startY)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
                trackers.append(tracker)
                tracker_boxes.append(bbox)
                tracker_labels.append(len(tracker_labels))
                if len(tracker_labels) == 2:
                    break
        if len(tracker_labels) > 0:
            tracker_initialized = True

    if len(person_boxes) == 0:
        tracker_initialized = False
        trackers.clear()
        tracker_labels.clear()
        arduino.write(b'STOP\n')
        target_idx = None
        last_command = "STOP"

    hand_map = {tracker_labels[i]: {"fist": 0, "open": 0, "one_finger": 0, "total_hands": 0, "five_fingers": 0}
                for i in range(len(person_boxes))}

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            wrist = handLms.landmark[0]
            hand_x = int(wrist.x * w)
            hand_y = int(wrist.y * h)

            closest_idx = None
            closest_dist = float('inf')
            for i, (x1, y1, x2, y2) in enumerate(person_boxes):
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                dist = math.hypot(hand_x - center_x, hand_y - center_y)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = tracker_labels[i]

            if closest_idx is not None:
                if is_fist(handLms):
                    hand_map[closest_idx]["fist"] += 1
                else:
                    hand_map[closest_idx]["open"] += 1
                if count_raised_fingers(handLms) == 1:
                    hand_map[closest_idx]["one_finger"] += 1
                if count_total_fingers(handLms) == 5:
                    hand_map[closest_idx]["five_fingers"] += 1
                hand_map[closest_idx]["total_hands"] += 1

    if target_idx is None:
        for pid, data in hand_map.items():
            if data["one_finger"] >= 1:
                target_idx = pid
                break

    if target_idx is not None and target_idx in hand_map:
        current_fingers = hand_map[target_idx]["one_finger"]
        if hand_map[target_idx]["total_hands"] >= 2 and current_fingers >= 2:
            for pid, data in hand_map.items():
                if pid == target_idx:
                    continue
                if data["total_hands"] >= 1 and data["one_finger"] == 1:
                    if switch_candidate_idx == pid:
                        switch_hold_frames += 1
                    else:
                        switch_candidate_idx = pid
                        switch_hold_frames = 1
                    if switch_hold_frames >= required_hold_frames and time.time() - last_switch_time > switch_cooldown:
                        target_idx = pid
                        last_switch_time = time.time()
                        switch_hold_frames = 0
                        switch_candidate_idx = None
                        break
        else:
            switch_candidate_idx = None
            switch_hold_frames = 0

    if target_idx is not None and target_idx in hand_map:
        data = hand_map[target_idx]
        if data["five_fingers"] >= 2:
            target_idx = None
            last_command = "STOP"
            arduino.write(b'STOP\n')

    if target_idx is not None and target_idx in hand_map:
        for i, tid in enumerate(tracker_labels):
            if tid == target_idx:
                (startX, startY, endX, endY) = person_boxes[i]
                midX = (startX + endX) // 2
                offset_x = midX - frame_center_x
                if data["fist"] == 2:
                    if offset_x < -50 and last_command != "LEFT":
                        arduino.write(b'LEFT\n')
                        last_command = "LEFT"
                    elif offset_x > 50 and last_command != "RIGHT":
                        arduino.write(b'RIGHT\n')
                        last_command = "RIGHT"
                    elif abs(offset_x) <= 50 and last_command != "FORWARD":
                        arduino.write(b'FORWARD\n')
                        last_command = "FORWARD"
                elif data["open"] == 2 and last_command != "STOP":
                    arduino.write(b'STOP\n')
                    last_command = "STOP"
                cv2.putText(frame, f"Target {tid} - Fists:{data['fist']} Open:{data['open']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Gesture + Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
