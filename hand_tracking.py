import cv2 as cv
import mediapipe as mp

vid_cap = cv.VideoCapture()
while vid_cap.isOpened():
    res, frame = cv.read()
    cv.imshow('Video Capture', frame)

    if cv.waitKey(10) & 0xff==ord('q'):
        break

vid_cap.release()
cv.destroyAllWindows()

mp_drawning = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

vid_cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while vid_cap.isOpened():
        res, frame = vid_cap.read()

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = holistic.process(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        mp_drawning.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawning.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv.imshow('Hand track', frame)

        if cv.waitKey(10) & 0xff==ord('q'):
            break

vid_cap.release()
cv.destroyAllWindows()