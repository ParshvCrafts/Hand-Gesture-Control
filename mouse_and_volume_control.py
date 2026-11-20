import cv2
import numpy as np
import time
import math
import autopy
import hand_tracking_module as htm
from pycaw.pycaw import AudioUtilities

# Camera settings - INCREASED SIZE (960x540 - 16:9 aspect ratio)
wCam, hCam = 2000, 1200

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector for 2 hands
detector = htm.HandDetector(max_num_hands=2, detection_confidence=0.7)

# Mouse control variables
prev_x, prev_y = 0, 0
current_x, current_y = 0, 0
screen_width, screen_height = autopy.screen.size()
smooth = 7

# Volume control variables
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

# FPS variables
pTime = 0

# Track current mode (for showing/hiding volume bar)
current_mode = None  # Can be "volume" or "mouse"

# Volume lock state - tracks if volume is being actively adjusted
volume_adjusting = False

# Mouse control area - SCALED PROPORTIONALLY
# Original proportions: left=70/640=0.109, right=570/640=0.891, top=50/480=0.104, bottom=330/480=0.688
mouse_frame_top = int(0.104 * hCam)      # ~56px (was 50)
mouse_frame_bottom = int(0.688 * hCam)   # ~371px (was 330)
mouse_frame_left = int(0.109 * wCam)     # ~105px (was 70)  
mouse_frame_right = int(0.891 * wCam)    # ~855px (was 570)

print("=== Virtual Mouse and Volume Control ===")
print("Left Hand: Volume Control")
print("  - Pinch gesture (Thumb + Index up, others down) = ADJUST volume")
print("  - Open any other finger = LOCK volume")
print("Right Hand: Mouse Control (Index finger) & Click (Index + Middle finger)")
print("Press 'q' to quit")
print("=" * 40)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flipping the frame horizontally
    img = cv2.flip(img, 1)  # Flip for mirror effect
    # Find all hands
    img = detector.findHands(img, draw=False)
    allHands = detector.findAllHands(img, draw=False)
    # Reset current mode
    current_mode = None
    # Process each detected hand
    for handNo, hand in enumerate(allHands):
        handType = hand["type"]
        lmList = hand["lmList"]
        
        # LEFT HAND (physical) - Volume Control
        # Note: MediaPipe reports "Right" for physical left hand when image is flipped
        if handType == "Left":
            current_mode = "volume"

            # Draw bounding box for left hand (green)
            detector.findBoundingBox(img, handNo=handNo, draw=True, offset=20, color=(255, 0, 255))

            # Get thumb tip (4) and index finger tip (8) positions
            x1, y1 = lmList[4]
            x2, y2 = lmList[8]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Check finger states for volume lock gesture
            # Volume only adjusts when: Thumb + Index are up, Middle + Ring + Pinky are down
            tipIds = [4, 8, 12, 16, 20]
            fingers = []
            
            # Thumb detection for Left hand (MediaPipe label after flip)
            # MediaPipe "Left" label with flipped image: thumb open when tip is RIGHT of IP joint
            if lmList[tipIds[0]][0] > lmList[tipIds[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other 4 fingers - tip above PIP joint means finger is up
            for id in range(1, 5):
                if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            # Pinch gesture: thumb up, index up, other fingers down
            # fingers = [thumb, index, middle, ring, pinky]
            is_pinch_gesture = (fingers[0] == 1 and fingers[1] == 1 and 
                               fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0)
            
            if is_pinch_gesture:
                volume_adjusting = True
                
                # Draw visual indicators (green = adjusting)
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                # Calculate distance
                length = math.hypot(x2 - x1, y2 - y1)

                # Change center circle color when fingers are close
                if length < 40:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

                # Map distance to volume (30-250 pixels -> minVol to maxVol)
                vol = np.interp(length, [30, 200], [minVol, maxVol])
                # SCALED volume bar position - maintain proportional positioning
                volBar = np.interp(length, [30, 200], [int(0.741 * hCam), int(0.278 * hCam)])  # ~400->150 scaled
                volPer = np.interp(length, [30, 200], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)

                # Display "ADJUSTING" label
                cv2.putText(img, "VOLUME: ADJUSTING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                volume_adjusting = False
                
                # Draw visual indicators (different color = locked)
                cv2.circle(img, (x1, y1), 10, (0, 165, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 165, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.circle(img, (cx, cy), 10, (0, 165, 255), cv2.FILLED)
                
                # Display "LOCKED" label - volume stays at current level
                cv2.putText(img, "VOLUME: LOCKED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # RIGHT HAND (physical) - Mouse Control
        # Note: MediaPipe reports "Left" for physical right hand when image is flipped
        elif handType == "Right":
            current_mode = "mouse"

            # Draw bounding box for right hand (magenta)
            detector.findBoundingBox(img, handNo=handNo, draw=True, offset=20, color=(255, 0, 255))

            # Get finger tips
            index_tip = lmList[8]
            middle_tip = lmList[12]
            x1, y1 = index_tip
            x2, y2 = middle_tip

            # Check which fingers are up
            fingers = detector.fingersUp(handNo=handNo)

            if len(fingers) == 5:
                # Draw mouse control area rectangle - USING SCALED COORDINATES
                cv2.rectangle(img, (mouse_frame_left, mouse_frame_top),
                            (mouse_frame_right, mouse_frame_bottom), (255, 0, 255), 2)

                # MOUSE MOVEMENT - Only index finger up
                if fingers[1] == 1 and fingers[2] == 0:
                    # Convert finger position to screen coordinates (NO flip for intuitive movement)
                    index_tip_x = np.interp(x1, (mouse_frame_left, mouse_frame_right), (0, screen_width))
                    index_tip_y = np.interp(y1, (mouse_frame_top, mouse_frame_bottom), (0, screen_height))

                    # Smooth the movement
                    current_x = prev_x + (index_tip_x - prev_x) / smooth
                    current_y = prev_y + (index_tip_y - prev_y) / smooth

                    # Move the mouse (direct mapping - no flip)
                    autopy.mouse.move(current_x, current_y)
                    cv2.circle(img, (x1, y1), 12, (0, 0, 255), cv2.FILLED)
                    prev_x, prev_y = current_x, current_y

                    # Display "MOUSE MODE" label
                    cv2.putText(img, "MOUSE MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # MOUSE CLICK - Both index and middle fingers up
                elif fingers[1] == 1 and fingers[2] == 1:
                    # Calculate distance between fingers
                    distance, info, _ = detector.findDistance(img, handNo=handNo, draw=True,
                                                    landmark_id_1=8, landmark_id_2=12)

                    # Click when fingers are close
                    if distance < 40:
                        cv2.putText(img, "CLICKING!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        autopy.mouse.click()
                        time.sleep(0.3)
                    else:
                        cv2.putText(img, "CLICK MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Volume Bar - SCALED PROPORTIONALLY (only visible in volume mode)
    if current_mode == "volume":
        # Original: (555, 150) to (610, 400) - maintaining proportions
        vol_bar_left = int(0.865 * wCam)    # ~830px (was 555)
        vol_bar_right = int(0.927 * wCam)   # ~890px (was 610)
        vol_bar_top = int(0.278 * hCam)     # ~150px (scaled)
        vol_bar_bottom = int(0.741 * hCam)  # ~400px (scaled)
        
        cv2.rectangle(img, (vol_bar_left, vol_bar_top), (vol_bar_right, vol_bar_bottom), (0, 255, 0), 3)
        cv2.rectangle(img, (vol_bar_left, int(volBar)), (vol_bar_right, vol_bar_bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(volPer)}%", (vol_bar_left, int(0.796 * hCam)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime > pTime else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, hCam - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Display the image
    cv2.imshow("Virtual Mouse and Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
