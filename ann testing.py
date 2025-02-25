import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_ann.h5')

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Label dictionary from the training phase
labels_dict = {0: 'down', 1: 'land', 2: 'left', 3: 'right', 4: 'up'}

# OpenCV Video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check if multiple hands are detected
    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        gestures = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            # Prepare the data for prediction
            x_ = []
            y_ = []
            data_aux = []

            # Collect x, y coordinates for landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the coordinates (relative to min/max values)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Predict the gesture class (use model for prediction)
            data_aux = np.asarray(data_aux).reshape(1, -1)  # Reshape for model input
            prediction = model.predict(data_aux)

            # Get the predicted class index
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = labels_dict[predicted_class]
            gestures.append(predicted_label)

            # Position the text next to each hand (different positions for each hand)
            y_position = 50 + idx * 100
            cv2.putText(frame, f'Hand {idx + 1}: {predicted_label}', (50, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Print the gestures of both hands in the console (optional)
        print("Detected Gestures: ", gestures)

    # Show the live video feed with predictions
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
