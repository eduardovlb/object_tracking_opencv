import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

ret, frame = cap.read()
frame_gray_init = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


parameters_lucas_kanade = dict(winSize = (15, 15), 
                                maxLevel = 2, 
                                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def select_point(event, x, y, flags, params):
    global point, selected_point, old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32)
cv.namedWindow('Frame')
cv.setMouseCallback('Frame', select_point)

selected_point = False
point = ()
old_points = np.array([[]])

mask = np.zeros_like(frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if selected_point is True:
        cv.circle(frame, point, 5, (0, 0, 255), 2)

        new_points, status, erros = cv.calcOpticalFlowPyrLK(frame_gray_init, 
                                                            frame_gray, 
                                                            old_points, 
                                                            None, 
                                                            **parameters_lucas_kanade)
        
        frame_gray_init = frame_gray.copy()
        old_points = new_points

        x, y = new_points.ravel()
        j, k = old_points.ravel()

        mask = cv.line(mask, (x, y), (j, k), (0, 255, 255), 2)
        frame = cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

    img = cv.add(frame, mask)

    cv.imshow("Frame", frame)
    cv.imshow("Frame2", mask)

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
cap.release()