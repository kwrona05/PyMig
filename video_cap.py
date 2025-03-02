#imports opencv and mediapipe
import cv2 as cv

#video capture
viedo = cv.VideoCapture(0)

#reading and showing captured video
while True:
    try:
        res, frame = viedo.read()
        cv.imshow("video", frame)
    except:
        pass
#abort program when 'q' key is down
    if cv.waitKey(1) & 0xff==ord('q'):
        break

cv.destroyAllWindows()

