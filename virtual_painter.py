import cv2 
import numpy as np  
import time
import os
import mediapipe as mp
import hand_tracking_module as htm

script_dir = os.path.dirname(os.path.abspath(__file__))
folderPath = os.path.join(script_dir, "assets", "toolbar")
myList = sorted(os.listdir(folderPath))  # 1-3 = red/blue/green, 4 = eraser
overlayList = []

for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    overlayList.append(image)


header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Get actual camera resolution
ret, test_frame = cap.read()
if not ret:
    raise SystemExit("No frame from webcam 0: check that a camera is connected and not in use by another app.")
frameHeight, frameWidth, _ = test_frame.shape

# Resize all overlay images to match camera width while preserving aspect ratio
overlayListResized = []
for overlay in overlayList:
    overlayHeight, overlayWidth, _ = overlay.shape
    aspectRatio = overlayHeight / overlayWidth
    newHeight = int(frameWidth * aspectRatio)

    # If new height is too tall, scale to fit 200px height instead
    if newHeight > 200:
        newHeight = 200
        newWidth = int(200 / aspectRatio)
        overlayResized = cv2.resize(overlay, (newWidth, newHeight))
        # Center it horizontally if narrower than frame
        if newWidth < frameWidth:
            centered = np.zeros((newHeight, frameWidth, 3), dtype=np.uint8)
            x_offset = (frameWidth - newWidth) // 2
            centered[:, x_offset:x_offset+newWidth] = overlayResized
            overlayResized = centered
    else:
        overlayResized = cv2.resize(overlay, (frameWidth, newHeight))

    overlayListResized.append(overlayResized)

header = overlayListResized[0]
headerHeight = header.shape[0]

# Calculate toolbar button positions dynamically
numButtons = len(overlayListResized)
buttonWidth = frameWidth // numButtons
buttonPositions = []
for i in range(numButtons):
    x_start = i * buttonWidth
    x_end = (i + 1) * buttonWidth
    buttonPositions.append((x_start, x_end, i))  # (start_x, end_x, index)

detector = htm.HandDetector(detection_confidence= 0.7)

draw_color = (255, 0, 255)
toolbar = {0: "Red", 1: "Blue", 2: "Green", 3: "Eraser"}
idx = None
pTime = 0

xp, yp = 0, 0
imgCanvas = np.zeros((frameHeight, frameWidth, 3), dtype=np.uint8)

while True:
    # Reading the frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Flipping the frame horizontally
    img = cv2.flip(img, 1)

    # Detecting the hand landmarks
    img = detector.findHands(img, draw=False)
    allHands = detector.findAllHands(img, draw=False)

    for handNo, hand in enumerate(allHands):  # both hands can act in the same frame
        lmList = hand["lmList"]
        # MediaPipe label on the mirrored frame = physical hand: "Right" paints, "Left" clears.
        if hand["type"] == "Right":
            # Index finger tip and middle finger tip
            x1, y1 = lmList[8]
            x2, y2 = lmList[12]

            # Check which fingers are up
            fingers = detector.fingersUp(handNo)

            # Selection Mode - If index and middle finger are up
            if fingers[1] and fingers[2]:
                cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), draw_color, cv2.FILLED)
                xp, yp = x1, y1

            # Checking for which toolbar button is pressed
            if y1 < headerHeight:
                # Check which button the finger is over
                for x_start, x_end, idx in buttonPositions:
                    if x_start < x1 < x_end:
                        header = overlayListResized[idx]
                        # print(f"Selected toolbar item {idx + 1}")
                        if idx == 0:
                            draw_color = (0, 0, 255)
                        elif idx == 1:
                            draw_color = (255, 0, 0)
                        elif idx == 2:
                            draw_color = (0, 255, 0)
                        elif idx == 3:
                            draw_color = (0, 0, 0)
    
                        break
            

            # Displaying the toolbar name
            cv2.rectangle(img, (0, 60), (100, 95), (255, 255, 255), cv2.FILLED)
            if idx is None:
                cv2.putText(img, "None", (7, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
            if idx == 0:
                cv2.putText(img, toolbar[idx], (7, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif idx == 1:
                cv2.putText(img, toolbar[idx], (7, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif idx == 2:
                cv2.putText(img, toolbar[idx], (7, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif idx == 3:
                cv2.putText(img, toolbar[idx], (7, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Drawing mode - If only index finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                if idx is not None:
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), draw_color, 15)
                    xp, yp = x1, y1
                
                if idx == 3:
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), draw_color, 50)
                    xp, yp = x1, y1

        # Clear all button
        if hand["type"] == "Left":  # left hand touching the toolbar clears the canvas
            x1, y1 = lmList[8]
            x2, y2 = lmList[12]
            fingers = detector.fingersUp(handNo)

            if fingers[1] and fingers[2]:
                cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), draw_color, cv2.FILLED)
                cv2.rectangle(img, (0, 60), (100, 95), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, "Clear", (7, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                if y1 < headerHeight:
                    # Check which button the finger is over
                    for x_start, x_end, _ in buttonPositions:  # not idx: clearing must keep the selected tool
                        if x_start < x1 < x_end:
                            imgCanvas = np.zeros((frameHeight, frameWidth, 3), dtype=np.uint8)
                            xp, yp = 0, 0
                            break


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the Overlay header image on the top of the frame
    img[0:headerHeight, 0:frameWidth] = header
    
    # Displaying the frame
    img = cv2.addWeighted(imgCanvas, 0.5, img, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()