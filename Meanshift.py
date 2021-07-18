import cv2 as cv
import time
from imutils.video import VideoStream

#cap = VideoStream(src=0).start()
#time.sleep(1.0)

cap = cv.VideoCapture(0)
ret, frame = cap.read()

bbox = cv.selectROI(frame, False)

x, y, w, h = bbox
track_window = (x, y, w, h)
#print(track_window)

roi = frame[y:y+h, x:x+w]
#cv.imshow('ROI', roi)
#cv.waitKey(0)

hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
#cv.imshow("ROI HSV", hsv_roi)
#cv.waitKey(0)

roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])

import matplotlib.pyplot as plt
plt.hist(roi.ravel(), 180, [0, 180])
plt.show()
#cv.waitKey(0)

roi_hist = cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0,180],1)
        ret, track_window = cv.meanShift(dst, (x, y, w, h), term_crit)

        x, y, w, h = track_window
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('Meanshift Tracking', frame)
        cv.imshow('dst', dst)
        cv.imshow('ROI', roi)

        if cv.waitKey(1) == 13:
            break
    else:
        break


cv.destroyAllWindows()
cap.release()