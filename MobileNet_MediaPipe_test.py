import cv2
import mediapipe as mp
import serial
import time
import math
import pygame
import threading

# ─────────────────────────────────────────────
# Serial communication with Arduino
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

# ─────────────────────────────────────────────
# Video + MediaPipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=6, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# Load MobileNet SSD for multi-person detection
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
# Helpers
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
    return 1 if index_up and middle_down and ring_down and pinky_down else 0

def count_total_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    return sum(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y for tip in tip_ids[1:])

def is_rock_out(hand_landmarks):
    def is_up(tip, pip):
        return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    index_up = is_up(8, 6)
    middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_up = is_up(20, 18)
    return index_up and pinky_up and middle_down and ring_down

pygame.mixer.init()
pygame.mixer.music.load('/home/owen/aiTankProj/songs/laUltimaNoche.mp3')  # replace with your song


# ─────────────────────────────────────────────
# Globals
last_command = None
last_switch_time = 0
switch_cooldown = 2
switch_candidate_idx = None
switch_hold_frames = 0
required_hold_frames = 15
target_idx = None
rockout_count = 0
last_rockout_time = 0
rockout_cooldown = 5
close_threshold = 450
is_playing = False
rockout_hold_start = None
rockout_gesture_active = False
rockout_start_time = None  # Track when the rock out gesture started


# ─────────────────────────────────────────────
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    frame_center_x = w // 2

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    person_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if CLASSES[class_id] == "person":
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                person_boxes.append((startX, startY, endX, endY))

    if len(person_boxes) == 0:
        print("👋 Everyone left the frame — resetting target")
        target_idx = None
        last_command = "STOP"
        arduino.write(b'STOP\n')

    for idx, (startX, startY, endX, endY) in enumerate(person_boxes):
        label = f"Person {idx + 1}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if target_idx is not None and target_idx < len(person_boxes):
        (startX, startY, endX, endY) = person_boxes[target_idx]
        offset_x = ((startX + endX) // 2) - frame_center_x
        target_height = endY - startY
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, "Target", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        offset_x = 0
        target_height = 0

    hand_map = {
        i: {"fist": 0, "open": 0, "one_finger": 0, "five_fingers": 0, "total_hands": 0, "raised_hands": 0}
        for i in range(len(person_boxes))
    }

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            wrist = handLms.landmark[0]
            hand_x = int(wrist.x * w)
            hand_y = int(wrist.y * h)

            closest_idx = None
            closest_dist = float('inf')
            for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                dist = math.hypot(hand_x - box_center_x, hand_y - box_center_y)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx

            if closest_idx is not None:
                head_y = (person_boxes[closest_idx][1] + person_boxes[closest_idx][3]) / 2
                if hand_y < head_y:
                    hand_map[closest_idx]["raised_hands"] += 1
                    hand_map[closest_idx]["total_hands"] += 1
                    hand_map[closest_idx]["fist"] += int(is_fist(handLms))
                    hand_map[closest_idx]["open"] += int(not is_fist(handLms))
                    hand_map[closest_idx]["one_finger"] += int(count_raised_fingers(handLms) == 1)
                    hand_map[closest_idx]["five_fingers"] += int(count_total_fingers(handLms) == 5)

            if is_rock_out(handLms):
                if rockout_start_time is None:
                    rockout_start_time = time.time()
                elif time.time() - rockout_start_time >= 2:
                    if not pygame.mixer.music.get_busy():
                        print("🎸 ROCK OUT detected – playing music!")
                        pygame.mixer.music.play()
                    else:
                        print("🛑 ROCK OUT held – stopping music.")
                        pygame.mixer.music.stop()
                    rockout_start_time = None  # reset so it won't repeat
                    break  # avoid double triggers from multiple hands
            else:
                rockout_start_time = None


    if target_idx is None:
        for idx, data in hand_map.items():
            if data["one_finger"] >= 1 and data["total_hands"] >= 1:
                print(f"🎯 Initial target set to Person {idx}")
                target_idx = idx
                break

    if target_idx is not None and target_idx in hand_map:
        if hand_map[target_idx]["total_hands"] >= 2 and hand_map[target_idx]["one_finger"] >= 2:
            for idx, data in hand_map.items():
                if idx == target_idx:
                    continue
                if data["total_hands"] >= 1 and data["one_finger"] == 1:
                    if switch_candidate_idx == idx:
                        switch_hold_frames += 1
                    else:
                        switch_candidate_idx = idx
                        switch_hold_frames = 1

                    if switch_hold_frames >= required_hold_frames and time.time() - last_switch_time > switch_cooldown:
                        print(f"🔁 Target switched from Person {target_idx} to Person {idx}")
                        target_idx = idx
                        last_switch_time = time.time()
                        switch_candidate_idx = None
                        switch_hold_frames = 0
                    break
        else:
            switch_candidate_idx = None
            switch_hold_frames = 0

 

    if target_idx is not None and target_idx in hand_map and hand_map[target_idx]["five_fingers"] >= 2:
        print("🛑 Target reset by waving both hands")
        target_idx = None
        last_command = "STOP"
        arduino.write(b'STOP\n')

    if target_idx is not None and target_idx in hand_map:
        raised = hand_map[target_idx]["raised_hands"]
        fists = hand_map[target_idx]["fist"]
        opens = hand_map[target_idx]["open"]

        if raised == 2:
            if fists == 2:
                if target_height >= close_threshold:
                    if last_command != "STOP":
                        arduino.write(b'STOP\n')
                        print("Sent: STOP (close to target)")
                        last_command = "STOP"
                else:
                    if offset_x < -50 and last_command != "LEFT":
                        arduino.write(b'LEFT\n')
                        print("Sent: LEFT")
                        last_command = "LEFT"
                    elif offset_x > 50 and last_command != "RIGHT":
                        arduino.write(b'RIGHT\n')
                        print("Sent: RIGHT")
                        last_command = "RIGHT"
                    elif abs(offset_x) <= 50 and last_command != "FORWARD":
                        arduino.write(b'FORWARD\n')
                        print("Sent: FORWARD")
                        last_command = "FORWARD"
            elif opens == 2 and last_command != "STOP":
                arduino.write(b'STOP\n')
                print("Sent: STOP")
                last_command = "STOP"
        else:
            print("Hands not raised – maintaining current state.")

    if target_idx is not None and target_idx in hand_map:
        cv2.putText(frame,
            f"Fists: {hand_map[target_idx]['fist']}, Opens: {hand_map[target_idx]['open']}, 1Fingers: {hand_map[target_idx]['one_finger']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No target detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Gesture + Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
