import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode = False, max_num_hands=2, detection_confidence=0.5, track_confidence = 0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.id_cx_cy = {}
        self.landmark_id_1 = 0
        self.landmark_id_2 = 0
        
        self.mpHands = mp.solutions.hands # Importing the hands module from mediapipe
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands=self.max_num_hands, min_detection_confidence=self.detection_confidence, min_tracking_confidence=self.track_confidence) 
        self.mpDraw = mp.solutions.drawing_utils # Importing the drawing utils module to draw the landmarks and connections
        self.green_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))

    def findHands(self, img, draw=True, landmark_id_1 = None, landmark_id_2 = None):
        self.landmark_id_1 = landmark_id_1
        self.landmark_id_2 = landmark_id_2
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting the image to RGB
        self.results = self.hands.process(imgRGB) # Processing the image to detect the hands, it outputs a class object
        # print(results.multi_hand_landmarks) 
        if draw:
            if self.results.multi_hand_landmarks: # Checking if the hand is detected from the class object
                for handLms in self.results.multi_hand_landmarks: # Looping through the hands
                    for id, lm in enumerate(handLms.landmark): # Looping through the landmarks, each id has a corresponding landmark x, y, z in decimal places
                        # However, the x and y coordinates are in decimal places, so we need to convert them to pixels
                        # print(id, lm)
                        h, w, c = img.shape # Getting the height, width and channels of the image
                        cx, cy = int(lm.x * w), int(lm.y * h) # Getting the x and y coordinates of the landmarks
                        # print(id, cx, cy) # Printing the id, x and y coordinates of the landmarks
                        # cx and cy allows to track the 21 landmarks
                        
                        if self.landmark_id_1 is not None:
                            if id == self.landmark_id_1:
                                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # Drawing a circle at the landmarks
                        if self.landmark_id_2 is not None:
                            if id == self.landmark_id_2:
                                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # Drawing a circle at the landmarks
                    
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, connection_drawing_spec= self.green_drawing_spec) # Drawing the landmarks and connections on the image which is displayed, not the imgRGB
        return img

    def findPosition(self, img, handNo=0, draw=True):
        handType = None
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            # Get hand type (Left or Right)
            if self.results.multi_handedness:
                handType = self.results.multi_handedness[handNo].classification[0].label  # "Left" or "Right"
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.id_cx_cy[id] = (cx, cy)
                if self.landmark_id_1 is not None:
                            if id == self.landmark_id_1:
                                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if self.landmark_id_2 is not None:
                            if id == self.landmark_id_2:
                                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
        return self.id_cx_cy, handType

def main():
    pTime = 0 
    cTime = 0
    cap = cv2.VideoCapture(0) # Accesing the front camera which is 0 in my case, errored for 1
    detector = HandDetector()
    while True:
        success, img = cap.read() # Capturing the video frame by frame
        
        img = detector.findHands(img, draw=True, landmark_id_1=8, landmark_id_2=None)
        id_cx_cy, handType = detector.findPosition(img, draw=True)
        if handType:
            print(f"Hand detected: {handType}")

        cTime = time.time() # Calculating the frames per second which is the time taken to process each frame
        fps = 1 / (cTime - pTime)
        pTime = cTime # Previous time started as 0 and then updated to the current time
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) # Displaying the fps on the image with scale 3, purple color and thickness 3
        
        cv2.imshow("Image", img) # Displaying the image
        if cv2.waitKey(1) & 0xFF == ord("q"): # Pressing q to exit the loop
            break

    # Releasing the camera and closing the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
