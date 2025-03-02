import mediapipe as mp
import cv2
import pyautogui

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 560)
cap.set(4, 400) 

# State variables
is_accelerating = False
is_braking = False
is_left = False
is_right = False

# Turn on MediaPipe
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)  # Mirror view
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = holistic.process(img_rgb)

        height, width, _ = img.shape
        y_mid = height // 2
        x_mid = width // 2
        pose = "Neutral"

        try:
            if results.pose_landmarks:
                right_hand = (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height
                )

                # Controls
                if right_hand[1] < y_mid:  # Hand is raised -> Accelerate
                    pose = "Accelerating"
                    if not is_accelerating:
                        pyautogui.keyDown("up")
                        is_accelerating = True
                    if is_braking:
                        pyautogui.keyUp("down")
                        is_braking = False

                elif right_hand[1] > y_mid:  # Hand is lowered -> Brake
                    pose = "Braking"
                    if not is_braking:
                        pyautogui.keyDown("down")
                        is_braking = True
                    if is_accelerating:
                        pyautogui.keyUp("up")
                        is_accelerating = False

                if right_hand[0] < x_mid:  # Move hand left -> Turn Left
                    pose = "Turn Left"
                    if not is_left:
                        pyautogui.keyDown("left")
                        is_left = True
                    if is_right:  # Stop turning right
                        pyautogui.keyUp("right")
                        is_right = False

                elif right_hand[0] > x_mid:  # Move hand right -> Turn Right
                    pose = "Turn Right"
                    if not is_right:
                        pyautogui.keyDown("right")
                        is_right = True
                    if is_left:  # Stop turning left
                        pyautogui.keyUp("left")
                        is_left = False

                # Reset keys if no movement detected
                else:
                    if is_accelerating:
                        pyautogui.keyUp("up")
                        is_accelerating = False
                    if is_braking:
                        pyautogui.keyUp("down")
                        is_braking = False
                    if is_left:
                        pyautogui.keyUp("left")
                        is_left = False
                    if is_right:
                        pyautogui.keyUp("right")
                        is_right = False

        except Exception as e:
            print(f"Error: {e}")

        # Display text and guides on the screen
        cv2.putText(img, pose, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.line(img, (0, y_mid), (width, y_mid), (255, 0, 255), 2)  # Horizontal line
        cv2.line(img, (x_mid, 0), (x_mid, height), (255, 0, 255), 2)  # Vertical line
        cv2.imshow('Game control', img)

        # Close camera on 'q' press
        if cv2.waitKey(1) == ord('q'):
            break

# End program
cap.release()
cv2.destroyAllWindows()