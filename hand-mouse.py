import cv2
import mediapipe as mp
import pyautogui
import math

class HandMouseController:
    def __init__(self):
        # Initialize Mediapipe Hands and PyAutoGUI
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        self.click_flag = False
        self.smooth_factor = 5  # Adjust for smoother mouse movements
        self.prev_x, self.prev_y = 0, 0

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def smooth_mouse_movement(self, current_x, current_y):
        """Smooth the mouse movement for better usability."""
        smooth_x = self.prev_x + (current_x - self.prev_x) / self.smooth_factor
        smooth_y = self.prev_y + (current_y - self.prev_y) / self.smooth_factor
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def process_frame(self, frame):
        """Process each frame to detect hand gestures and control the mouse."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        landmarks = results.multi_hand_landmarks

        if landmarks:
            for hand_landmarks in landmarks:
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

                # Convert to screen coordinates
                index_pos = (int(index_tip.x * self.screen_width), int(index_tip.y * self.screen_height))
                thumb_pos = (int(thumb_tip.x * self.screen_width), int(thumb_tip.y * self.screen_height))

                # Move the mouse pointer smoothly
                smoothed_pos = self.smooth_mouse_movement(*index_pos)
                pyautogui.moveTo(smoothed_pos[0], smoothed_pos[1])

                # Calculate distance for pinch detection
                distance = self.calculate_distance(index_pos, thumb_pos)
                cv2.circle(frame, index_pos, 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, thumb_pos, 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f'Distance: {int(distance)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Click if pinch is detected
                if distance < 50:
                    if not self.click_flag:
                        pyautogui.click()
                        self.click_flag = True
                else:
                    self.click_flag = False

                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def run(self):
        """Main loop to capture frames and process gestures."""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            processed_frame = self.process_frame(frame)
            cv2.imshow("Hand Mouse Controller", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandMouseController()
    controller.run()
