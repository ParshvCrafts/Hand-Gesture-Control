import cv2
import numpy as np
import time
import os
import mediapipe as mp
import autopy
import hand_tracking_module as htm
import time

wCam, hCam = 1200, 1080

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize the hand detector
detector = htm.HandDetector(max_num_hands=1)

# Initialize the previous position of the mouse
prev_x, prev_y = 0, 0

# Initialize the current position of the mouse
current_x, current_y = 0, 0

# Initialize the frame rate
current_frame = 0
previous_frame = 0

screen_width, screen_height = autopy.screen.size()
smooth = 7

while True:
    # Read the frame
    success, img = cap.read()
    if not success:
        break

    # Find the hand landmarks
    img = detector.findHands(img, draw=True)
    hands = detector.findAllHands(img, draw=False)
    id_cx_cy = hands[0]["lmList"] if hands else {}  # {id: (x, y)}
    bboxs = detector.findBoundingBox(img, draw= True, color = (0, 255, 0))

    # Getting the tip of the index and middle finger
    if id_cx_cy:
        index_tip = id_cx_cy[8]
        middle_tip = id_cx_cy[12]

        cv2.rectangle(img, (70, 70), (wCam - 130, hCam - 130), (255, 0, 255), 2)

        fingers = detector.fingersUp()
        if len(fingers) == 5:
            # Only Index Finger: Moving mouse mode
            if fingers[1] == 1 and fingers[2] == 0:
                
                # Convert the tip of the index and middle finger to screen coordinates
                index_tip_x = np.interp(index_tip[0], (70, wCam - 130), (0, screen_width))
                index_tip_y = np.interp(index_tip[1], (70, hCam - 130), (0, screen_height))
                
                # Smoothing the value
                current_x = prev_x + (index_tip_x - prev_x) / smooth
                current_y = prev_y + (index_tip_y - prev_y) / smooth

                # Move the mouse
                autopy.mouse.move(screen_width - current_x, current_y)
                cv2.circle(img, (index_tip[0], index_tip[1]), 15, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = current_x, current_y


            # Both Index and Middle Finger: Clicking mouse mode
            if fingers[1] == 1 and fingers[2] == 1:
                # If the distance is short, click
                distance, _, _ = detector.findDistance(img, handNo=0, draw=True, landmark_id_1=8, landmark_id_2=12)
                
                if distance < 40:
                    autopy.mouse.click()
                    time.sleep(0.2)

    # Calculate the frame rate
    current_frame = time.time()
    fps = 1 / (current_frame - previous_frame)
    previous_frame = current_frame

    # Display the frame rate
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

