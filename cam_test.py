import cv2

# Desired display size (fullscreen-ish, or set dynamically later)
display_width = 1280
display_height = 720

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

# Create a resizable window
cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Feed", display_width, display_height)

print("✅ Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize the frame to fit the display window
    resized_frame = cv2.resize(frame, (display_width, display_height))

    cv2.imshow("Webcam Feed", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
