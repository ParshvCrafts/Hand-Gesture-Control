import cv2
import mediapipe as mp
import time
import numpy as np
import hand_tracking_module as htm
import math
from pycaw.pycaw import AudioUtilities

#wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
#cap.set(3, wCam)
# cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector( detection_confidence= 0.6 )
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
volRange = volume.GetVolumeRange()
print(volRange)
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.findHands(img)
    hands = detector.findAllHands(img, draw=False)
    lmList = hands[0]["lmList"] if hands else {}  # {id: (x, y)}
    if lmList:
        # print(lmList[4])
        x1, y1 = lmList[4][0], lmList[4][1]
        x2, y2 = lmList[8][0], lmList[8][1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)
        if length < 40:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Hand range 30 - 250
        # Volume range -65.25 to 0.0
        # Mapping the length to the volume
        vol = np.interp(length, [30, 250], [minVol, maxVol])
        volBar = np.interp(length, [30, 250], [500, 150])
        volPer = np.interp(length, [30, 250], [0, 100])
        #print(int(length),int(vol))
        volume.SetMasterVolumeLevel(vol, None)

    # Volume Bar
    cv2.rectangle(img, (30, 150), (85, 500), (0, 255, 0), 3)
    cv2.rectangle(img, (30, int(volBar)), (85, 500), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"Volume: {int(volPer)} %", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime > pTime else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # Displaying the fps on the image with scale 3, purple color and thickness 3
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Releasing the camera and closing the window
cap.release()
cv2.destroyAllWindows()