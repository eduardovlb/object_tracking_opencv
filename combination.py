import cv2 as cv
import sys
from random import randint

tracker = cv.TrackerCSRT_create()

video = cv.VideoCapture('videos/walking.avi')

if not video.isOpened():
    print("Não foi possível abrir o vídeo")
    sys.exit()

ok, frame = video.read()
if not ok:
    print("Não é possível abrir o arquivo de vídeo")
    sys.exit()

cascade = cv.CascadeClassifier('cascade/fullbody.xml')

def detectar():
    while True:
        ok, frame = video.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detection = cascade.detectMultiScale(frame_gray)

        for (x, y, l, a) in detection:
            cv.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
            cv.imshow("Detection", frame)

            #cv.waitKey(0)
            #cv.destroyAllWindows()

            if x > 0:
                print("Detecção efetuada pelo haarcascade")
                return x, y, l, a

            
bbox = detectar()
#print(bbox)

ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv.rectangle(frame, (x, y), (x + w, y+ h), colors, 2, 1)
    else:
        print("Falha no rastreamento. Será executado o deector haarcascade")
        bbox = detectar()
        tracker = cv.TrackerCSRT_create()

    cv.imshow("Tracking", frame)
    k = cv.waitKey(1) & 0XFF
    if k == 27:
        break

print("OK")