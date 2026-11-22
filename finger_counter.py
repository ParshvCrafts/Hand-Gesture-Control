import cv2
import mediapipe as mp
import time
import numpy as np
import hand_tracking_module as htm
import os

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
folderPath = os.path.join(script_dir, "assets", "fingers")
myList = sorted(os.listdir(folderPath))  # 1..9 = that many fingers, 91 = ten, 92 = zero
overlayList = []

for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    overlayList.append(image)

detector = htm.HandDetector(detection_confidence= 0.6)

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = detector.findHands(img)
    allHands = detector.findAllHands(img, draw=False)
    
    totalFingers = 0
    
    if allHands:
        # Loop through each detected hand
        for hand in allHands:
            # Unflipped frame: MediaPipe's label is the opposite physical hand, but fingers_up's thumb rule
            # works on the label either way (label and geometry mirror together).
            totalFingers += sum(htm.fingers_up(hand["lmList"], hand["type"]))
        
        # Display total finger count
        if totalFingers == 10:
            cv2.rectangle(img, (20, 225), (240, 425), (0, 255, 0), cv2.FILLED)
        else:
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        
        cv2.putText(img, str(totalFingers), (60, 375), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 10)
        
        # Show overlay image based on total fingers
        if totalFingers == 0:
            # Show image 92 (0 fingers)
            overlayIndex = len(overlayList) - 1  # Last image (92)
        elif totalFingers <= 10:
            overlayIndex = totalFingers - 1
        else:
            overlayIndex = None
            
        if overlayIndex is not None and overlayIndex < len(overlayList):
            h, w, c = overlayList[overlayIndex].shape
            img[0:h, 0:w] = overlayList[overlayIndex]
    
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime > pTime else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (700, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # Displaying the fps on the image with scale 3, purple color and thickness 3
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Releasing the camera and closing the window
cap.release()

cv2.destroyAllWindows()