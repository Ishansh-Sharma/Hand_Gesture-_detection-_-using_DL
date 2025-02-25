import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

gestures = ['up', 'down', 'left', 'right', 'land']  # Gesture names
dataset_size = 30

cap = cv2.VideoCapture(0)

for gesture in gestures:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    print(f'Get ready for gesture: {gesture}')
    time.sleep(3)
    print(f'Starting data collection for "{gesture}".')

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Camera not accessible")
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Capturing {gesture}: {counter + 1}/{dataset_size}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # Save the image to the respective directory
        cv2.imwrite(os.path.join(gesture_dir, f'{counter}.jpg'), frame)

        counter += 1
        time.sleep(2)  # Wait for 2 seconds between captures

cap.release()
cv2.destroyAllWindows()
