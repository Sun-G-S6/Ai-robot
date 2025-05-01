import cv2
import mediapipe as mp
import serial
import time
import math

from collections import deque

# ─────────────────────────────────────────────
# Serial communication with Arduino
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

# ─────────────────────────────────────────────
# Video + MediaPipe
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=3)
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# Globals and State
last_command = None
last_switch_time = 0
switch_cooldown = 2  # seconds
switch_candidate_idx = None
switch_hold_frames = 0
required_hold_frames = 15

target_idx = None
pose_boxes = []

# ─────────────────────────────────────────────
# Gesture Helpers
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

# ─────────────────────────────────────────────
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    frame_center_x = w // 2

    pose_results = pose.process(frame_rgb)
    hands_results = hands.process(frame_rgb)

    # Detect people using pose landmarks
    pose_boxes = []
    if pose_results.pose_landmarks:
        xs = [lm.x for lm in pose_results.pose_landmarks.landmark]
        ys = [lm.y for lm in pose_results.pose_landmarks.landmark]
        x_min, x_max = int(min(xs) * w), int(max(xs) * w)
        y_min, y_max = int(min(ys) * h), int(max(ys) * h)
        pose_boxes.append((x_min, y_min, x_max, y_max))
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Person disappears
    if len(pose_boxes) == 0:
        print("👋 Everyone left the frame — resetting target")
        target_idx = None
        last_command = "STOP"
        arduino.write(b'STOP\n')

    # Draw all persons
    for idx, (startX, startY, endX, endY) in enumerate(pose_boxes):
        label = f"Person {idx + 1}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Maintain target box and offset
    if target_idx is not None and target_idx < len(pose_boxes):
        (startX, startY, endX, endY) = pose_boxes[target_idx]
        offset_x = ((startX + endX) // 2) - frame_center_x
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, "Target", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        offset_x = 0

    # Associate hands to person
    hand_map = {i: {"fist": 0, "open": 0, "one_finger": 0, "five_fingers": 0, "total_hands": 0} for i in range(len(pose_boxes))}

    if hands_results.multi_hand_landmarks:
        for handLms in hands_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            wrist = handLms.landmark[0]
            hand_x = int(wrist.x * w)
            hand_y = int(wrist.y * h)

            closest_idx = None
            closest_dist = float('inf')
            for idx, (x1, y1, x2, y2) in enumerate(pose_boxes):
                if x1 <= hand_x <= x2 and y1 <= hand_y <= y2:
                    closest_idx = idx
                    break
                else:
                    box_center_x = (x1 + x2) // 2
                    box_center_y = (y1 + y2) // 2
                    dist = math.hypot(hand_x - box_center_x, hand_y - box_center_y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = idx

            if closest_idx is not None:
                hand_map[closest_idx]["total_hands"] += 1
                hand_map[closest_idx]["fist"] += int(is_fist(handLms))
                hand_map[closest_idx]["open"] += int(not is_fist(handLms))
                hand_map[closest_idx]["one_finger"] += int(count_raised_fingers(handLms) == 1)
                hand_map[closest_idx]["five_fingers"] += int(count_total_fingers(handLms) == 5)

    # Initial target assignment
    if target_idx is None:
        for idx, data in hand_map.items():
            if data["one_finger"] >= 1 and data["total_hands"] >= 1:
                print(f"🎯 Initial target set to Person {idx}")
                target_idx = idx
                break

    # Target switching logic
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

    # Reset gesture
    if target_idx is not None and hand_map[target_idx]["five_fingers"] >= 2:
        print("🛑 Target reset by waving both hands")
        target_idx = None
        last_command = "STOP"
        arduino.write(b'STOP\n')

    # Movement control
    if target_idx is not None:
        fists = hand_map[target_idx]["fist"]
        opens = hand_map[target_idx]["open"]
        if fists == 2:
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

    # Overlay info
    if target_idx is not None and target_idx in hand_map:
        cv2.putText(frame,
            f"Fists: {hand_map[target_idx]['fist']}, Opens: {hand_map[target_idx]['open']}, 1Fingers: {hand_map[target_idx]['one_finger']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No target detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show
    cv2.imshow("Gesture + Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
arduino.close()
cv2.destroyAllWindows()
