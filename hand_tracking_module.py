"""
Hand Tracking Module
====================
Shared MediaPipe Hands wrapper used by every script in this repo.

- findHands() runs detection once per frame and caches each hand as
  {"type", "lmList" {id: (x, y)}, "handNo", "mp_label", "wrist_pos"}; findAllHands() returns that cache.
- "type" is MediaPipe's handedness label ("Left"/"Right"), whitespace-stripped. MediaPipe assumes a
  mirrored/selfie frame, so on a cv2.flip(img, 1) frame the label names the physical hand.
- fingersUp(handNo) / fingersUpForHand(hand) / fingers_up(lmList, type) return [thumb, index, middle, ring, pinky]
  as 0/1.
"""

import cv2
import mediapipe as mp
import math
from typing import List, Dict, Optional, Tuple

TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


def fingers_up(lm_list: Dict[int, Tuple[int, int]], hand_type: str) -> List[int]:
    """[thumb, index, middle, ring, pinky] as 0/1 for one hand, palm facing the camera.

    Thumb: MediaPipe's label describes the hand as it looks in the (assumed mirrored) image, so a hand labelled
    "Right" has its thumb on the image-left side: extended means tip x < IP-joint x ("Left" is the reverse).
    This holds on flipped and unflipped frames alike because label and geometry mirror together.
    Other fingers: up when the tip is above (smaller y than) the PIP joint.
    """
    tip_x, ip_x = lm_list[TIP_IDS[0]][0], lm_list[TIP_IDS[0] - 1][0]
    thumb = tip_x < ip_x if hand_type == "Right" else tip_x > ip_x
    others = [lm_list[tip][1] < lm_list[tip - 2][1] for tip in TIP_IDS[1:]]
    return [int(thumb)] + [int(f) for f in others]


class HandDetector:
    """MediaPipe Hands wrapper: run once per frame with findHands(), then query the cached hands."""
    
    def __init__(self, mode=False, max_num_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        
        # MediaPipe setup
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.track_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self._hands_cache = []
        
        # Drawing specs
        self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=2, circle_radius=2, color=(0, 255, 0)
        )
        self.connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=2, color=(0, 255, 0)
        )
        
        # Finger tip landmark IDs
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        # Store the last processed results and image dimensions
        self.results = None
        self.img_height = 0
        self.img_width = 0
    

    def findHands(self, img, draw=False):
        """
        Detect hands in the image and optionally draw landmarks.
        Builds internal cache for findAllHands().
        """
        self.img_height, self.img_width = img.shape[:2]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Clear and rebuild cache
        self._hands_cache = []

        if self.results.multi_hand_landmarks:
            for hand_idx, handLms in enumerate(self.results.multi_hand_landmarks):
                # Extract landmarks
                landmarks = {}
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * self.img_width), int(lm.y * self.img_height)
                    landmarks[id] = (cx, cy)

                # Get wrist position
                wrist_x, wrist_y = landmarks[0]

                # Determine hand type based on MediaPipe's classification
                # strip(): the Windows mediapipe 0.10.8 wheel ships handedness.txt with CRLF line endings, so
                # labels arrive as "Right\r" and every == "Right" comparison silently fails without it.
                mp_label = "Right"  # Default
                if self.results.multi_handedness:
                    mp_label = self.results.multi_handedness[hand_idx].classification[0].label.strip()

                # Create hand info dict - use MediaPipe's label directly
                hand_info = {
                    "type": mp_label,  # Use MediaPipe's label directly
                    "lmList": landmarks,
                    "handNo": hand_idx,
                    "mp_label": mp_label,
                    "wrist_pos": (wrist_x, wrist_y)
                }

                self._hands_cache.append(hand_info)

                # Draw if requested
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.landmark_drawing_spec,
                        self.connection_drawing_spec
                    )

        return img

    def findAllHands(self, img, draw=True) -> List[Dict]:
        """
        Returns a list of all detected hands with their landmarks and hand type.

        Returns:
            List of dictionaries, each containing:
                - "type": "Left" or "Right" (the PHYSICAL hand)
                - "lmList": Dictionary of landmarks {id: (x, y)}
                - "handNo": Index in MediaPipe results
                - "mp_label": What MediaPipe reported (for debugging)
                - "wrist_pos": Tuple (x, y) of wrist position
        """
        return self._hands_cache

    def findPosition(self, img, handNo=0, draw=False):
        self.lmList = []
        xList = []
        yList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self, handNo=0):
        """
        Determine which fingers are up for a specific hand.

        Args:
            handNo: Index of the hand (0 for first detected hand, 1 for second)

        Returns:
            List of 5 integers [thumb, index, middle, ring, pinky] where 1=up, 0=down
        """
        if handNo >= len(self._hands_cache):
            return [0, 0, 0, 0, 0]

        hand_info = self._hands_cache[handNo]
        return fingers_up(hand_info["lmList"], hand_info["type"])

    def fingersUpForHand(self, hand_dict: Dict) -> List[int]:
        """
        Determine which fingers are up for a hand given its dictionary.
        
        This is a convenience method that takes the hand dictionary
        returned by findAllHands() directly.
        
        Args:
            hand_dict: Dictionary from findAllHands() containing 'handNo'
        
        Returns:
            List of 5 integers [thumb, index, middle, ring, pinky]
        """
        hand_no = hand_dict.get("handNo", 0)
        return self.fingersUp(handNo=hand_no)

    def findBoundingBox(self, img, handNo=0, draw=True, offset=20, color=(255, 0, 255)):
        """
        Find and optionally draw a bounding box around a detected hand.

        Returns:
            Tuple (xmin, ymin, width, height) or None if hand not found
        """
        if handNo >= len(self._hands_cache):
            return None

        hand_info = self._hands_cache[handNo]
        landmarks = hand_info["lmList"]
        
        x_coords = [pos[0] for pos in landmarks.values()]
        y_coords = [pos[1] for pos in landmarks.values()]
        
        xmin = max(0, min(x_coords) - offset)
        ymin = max(0, min(y_coords) - offset)
        xmax = min(self.img_width, max(x_coords) + offset)
        ymax = min(self.img_height, max(y_coords) + offset)
        
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        
        if draw:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            # Corner markers
            cv2.circle(img, (xmin, ymin), 5, color, cv2.FILLED)
            cv2.circle(img, (xmax, ymin), 5, color, cv2.FILLED)
            cv2.circle(img, (xmin, ymax), 5, color, cv2.FILLED)
            cv2.circle(img, (xmax, ymax), 5, color, cv2.FILLED)
        
        return bbox

    def findDistance(self, img, handNo=0, landmark_id_1=8, landmark_id_2=12, draw=False, r=15, t=3):
        """
        Find distance between two landmarks on a specific hand.

        Args:
            img: Image to draw on
            handNo: Index of the hand
            landmark_id_1: First landmark ID (default 8 = index finger tip)
            landmark_id_2: Second landmark ID (default 12 = middle finger tip)
            draw: Whether to draw the line and circles
            r: Radius of circles
            t: Thickness of line

        Returns:
            Tuple (length, info, img) where info is [x1, y1, x2, y2, cx, cy]
        """
        if handNo >= len(self._hands_cache):
            return 0, [0, 0, 0, 0, 0, 0], img

        hand_info = self._hands_cache[handNo]
        lm_list = hand_info["lmList"]

        if landmark_id_1 not in lm_list or landmark_id_2 not in lm_list:
            return 0, [0, 0, 0, 0, 0, 0], img

        x1, y1 = lm_list[landmark_id_1]
        x2, y2 = lm_list[landmark_id_2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, [x1, y1, x2, y2, cx, cy], img

    def getHandCenter(self, handNo=0) -> Optional[Tuple[int, int]]:
        """Get the center point of a hand (average of all landmarks)."""
        if handNo >= len(self._hands_cache):
            return None

        landmarks = self._hands_cache[handNo]["lmList"]
        if not landmarks:
            return None
        
        avg_x = sum(pos[0] for pos in landmarks.values()) // len(landmarks)
        avg_y = sum(pos[1] for pos in landmarks.values()) // len(landmarks)
        
        return (avg_x, avg_y)


def main():
    """Test the hand detector with visual feedback."""
    import time
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    detector = HandDetector(detection_confidence=0.7)
    
    pTime = 0
    
    print("=" * 60)
    print("HAND TRACKING MODULE TEST")
    print("=" * 60)
    print("Raise your hands to test detection")
    print("LEFT hand should show 'Left', RIGHT hand should show 'Right'")
    print("Press 'q' to quit")
    print("=" * 60)
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Flip for mirror effect
        img = cv2.flip(img, 1)
        
        # Detect hands
        img = detector.findHands(img, draw=True)
        all_hands = detector.findAllHands(img, draw=False)
        
        # Process each hand
        for hand in all_hands:
            hand_type = hand["type"]
            hand_no = hand["handNo"]
            wrist_x, wrist_y = hand["wrist_pos"]
            
            # Get fingers up
            fingers = detector.fingersUp(handNo=hand_no)
            
            # Draw bounding box with color based on hand type
            color = (255, 0, 0) if hand_type == "Left" else (255, 0, 0)
            detector.findBoundingBox(img, handNo=hand_no, draw=True, color=color)
            
            # Display hand type and finger count
            finger_count = sum(fingers) if fingers else 0
            text = f"{hand_type}: {finger_count} fingers"
            cv2.putText(img, text, (wrist_x - 50, wrist_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Debug info
            mp_label = hand.get("mp_label", "?")
            debug_text = f"MP:{mp_label}"
            # cv2.putText(img, debug_text, (wrist_x - 50, wrist_y - 25),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        # cv2.putText(img, "GREEN = Left Hand | BLUE = Right Hand", (10, 470),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Hand Tracking Test", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()