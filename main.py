import cv2
import mediapipe as mp
import serial
import time
import math

from collections import deque

trackers = cv2.MultiTracker_create()
tracker_initialized = False
tracker_labels = []  # Maintain logical IDs
tracker_history = deque(maxlen=5)  # Optional smoothing

last_switch_time = 0
switch_cooldown = 2  # seconds
switch_candidate_idx = None
switch_hold_frames = 0
required_hold_frames = 15


# ─────────────────────────────────────────────
# Helper: Detect if hand is closed (fist)
def is_fist(hand_landmarks):
    # Tip of thumb (4), tip of index (8)
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return dist < 0.05

# ─────────────────────────────────────────────
# Helper: Count raised fingers for transfer of ownership
def count_raised_fingers(hand_landmarks):
    # Only index finger up (8 above 6), others not raised
    def is_up(tip, pip):
        return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y

    index_up = is_up(8, 6)
    middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y

    if index_up and middle_down and ring_down and pinky_down:
        return 1
    return 0


# ─────────────────────────────────────────────
# Helper: Count total fingers for transfer of ownership
def count_total_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    count = 0
    for tip in tip_ids[1:]:  # skip thumb for simplicity
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count


# ─────────────────────────────────────────────
# Init serial connection to Arduino
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

# ─────────────────────────────────────────────
# Init video capture
cap = cv2.VideoCapture(0)

# Init MobileNet-SSD (person detection)
model_dir = "mobilenet-ssd/"
net = cv2.dnn.readNetFromCaffe(
    model_dir + "MobileNetSSD_deploy.prototxt",
    model_dir + "MobileNetSSD_deploy.caffemodel"
)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Init MediaPipe (hand detection)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=3)
mp_draw = mp.solutions.drawing_utils

last_command = None
target_idx = None  # Global index of current target

# ─────────────────────────────────────────────
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    success, boxes = trackers.update(frame)

    # May or may not go here
    if success:
        person_boxes = []
        for i, newbox in enumerate(boxes):
            (x, y, w_box, h_box) = [int(v) for v in newbox]
            person_boxes.append((x, y, x + w_box, y + h_box))
            label = f"Person {tracker_labels[i]}"
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


    # ───── Person Detection ─────
    (h, w) = frame.shape[:2]
    frame_center_x = w // 2
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    #person detection
    person_detected = False

    if not tracker_initialized:
        person_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if CLASSES[class_id] == "person":
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (startX, startY, endX, endY) = box.astype("int")
                    bbox = (startX, startY, endX - startX, endY - startY)
                    person_boxes.append(bbox)
                    tracker = cv2.TrackerCSRT_create()  # Or KCF/MOSSE
                    trackers.add(tracker, frame, bbox)
                    tracker_labels.append(len(tracker_labels))
                    if len(person_boxes) == 2:
                        break
        if len(person_boxes) > 0:
            tracker_initialized = True


    # Extra Step: Reset target if no people in frame
    if len(person_boxes) == 0:
        print("👋 Everyone left the frame — resetting target")
        target_idx = None
        target_box = None
        last_command = "STOP"
        arduino.write(b'STOP\n')
        tracker_initialized = False
        trackers = cv2.MultiTracker_create()
        tracker_labels = []


    # Step 2: Draw boxes for both people
    for idx, (startX, startY, endX, endY) in enumerate(person_boxes):
        person_id = tracker_labels[idx]
        label = f"Person {idx + 1}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Step 3: Determine current target_box based on remembered target_idx
    if target_idx is not None and target_idx < len(person_boxes):
        target_box = person_boxes[target_idx]
        (startX, startY, endX, endY) = target_box
        offset_x = ((startX + endX) // 2) - frame_center_x
        person_detected = True


    # Step 4: Highlight the target
    if person_detected and target_box:
        (startX, startY, endX, endY) = target_box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, "Target", (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




  # ───── Hand Detection with Person Association ─────
    results = hands.process(frame_rgb)

    # Store hand counts per person (by index in person_boxes)
    hand_map = {i: {"fist": 0, "open": 0, "one_finger": 0, "total_hands": 0, "five_fingers": 0} for i in range(len(person_boxes))}


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Use wrist as hand location
            wrist = handLms.landmark[0]
            hand_x = int(wrist.x * w)
            hand_y = int(wrist.y * h)

            # Determine if it's a fist
            is_f = is_fist(handLms)

            # Associate this hand with the closest person box
            closest_idx = None
            closest_dist = float('inf')
            for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
                if x1 <= hand_x <= x2 and y1 <= hand_y <= y2:
                    closest_idx = idx
                    break  # hand inside box = strong match
                else:
                    # Otherwise, use center distance to box
                    box_center_x = (x1 + x2) // 2
                    box_center_y = (y1 + y2) // 2
                    dist = math.hypot(hand_x - box_center_x, hand_y - box_center_y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = idx

            # Record gesture for that person
            if closest_idx is not None:
                hand_map[closest_idx]["total_hands"] += 1

                if is_f:
                    hand_map[closest_idx]["fist"] += 1
                else:
                    hand_map[closest_idx]["open"] += 1

                # Count index finger gesture
                if count_raised_fingers(handLms) == 1:
                    hand_map[closest_idx]["one_finger"] += 1

                # Count full open palm (5 fingers)
                if count_total_fingers(handLms) == 5:
                    hand_map[closest_idx]["five_fingers"] += 1

    # ───── Initial Target Selection (if no one is set) ─────
    if target_idx is None:
        for idx, data in hand_map.items():
            if data["one_finger"] >= 1 and data["total_hands"] >= 1:
                print(f"🎯 Initial target set to Person {idx}")
                target_idx = idx
                target_box = person_boxes[target_idx]
                (startX, startY, endX, endY) = target_box
                offset_x = ((startX + endX) // 2) - frame_center_x
                person_detected = True
                break


    # ───── Command Logic ─────
    # Determine index of the target person in person_boxes
    # This resets the current target every loop, currently removed
    # target_idx = None
    # if target_box:
    #     for i, box in enumerate(person_boxes):
    #         if box == target_box:
    #             target_idx = i
    #             break

    # ───── Target Switching Logic ─────
    # If current target raises one finger on both hands,
    # and someone else raises one finger, allow switch.

    if target_idx is not None and target_idx in hand_map:
        current_target_fingers = hand_map[target_idx]["one_finger"]
        if hand_map[target_idx]["total_hands"] >= 2 and current_target_fingers >= 2:
            for idx, data in hand_map.items():
                if idx == target_idx:
                    continue
                if data["total_hands"] >= 1 and data["one_finger"] == 1:
                    # Check if same candidate is being considered
                    if switch_candidate_idx == idx:
                        switch_hold_frames += 1
                    else:
                        switch_candidate_idx = idx
                        switch_hold_frames = 1

                    # Confirm switch after N frames of consistent intent
                    if switch_hold_frames >= required_hold_frames and time.time() - last_switch_time > switch_cooldown:
                        print(f"🔁 Target switched from Person {target_idx} to Person {idx}")
                        target_idx = idx
                        target_box = person_boxes[target_idx]
                        (startX, startY, endX, endY) = target_box
                        offset_x = ((startX + endX) // 2) - frame_center_x
                        last_switch_time = time.time()
                        switch_candidate_idx = None
                        switch_hold_frames = 0
                    break
        # Clear switch candidate if conditions are not active
        if current_target_fingers < 2:
            switch_candidate_idx = None
            switch_hold_frames = 0




    # ───── Reset Target Logic ─────
    if target_idx is not None and target_idx in hand_map:
        if hand_map[target_idx]["five_fingers"] >= 2:
            print("🛑 Target reset by waving both hands")
            target_idx = None
            target_box = None
            last_command = "STOP"
            arduino.write(b'STOP\n')


    # Command Logic (stop/go/left/right/forward)
    if person_detected and target_idx is not None and target_idx in hand_map:
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



   # ───── Show Feed ─────
    if target_idx is not None and target_idx in hand_map:
        cv2.putText(frame,
            f"Fists: {hand_map[target_idx]['fist']}, Opens: {hand_map[target_idx]['open']}, 1Fingers: {hand_map[target_idx]['one_finger']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame,
            "No target detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Gesture + Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# ─────────────────────────────────────────────
# Cleanup
cap.release()
arduino.close()
cv2.destroyAllWindows()
