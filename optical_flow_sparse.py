import cv2 as cv
import numpy as np

cap = cv.VideoCapture('videos/walking.avi')

parameters_shitomasi = dict(maxCorners = 100, 
                            qualityLevel = 0.3, 
                            minDistance = 7)
parameters_lucas_kanade = dict(winSize = (15, 15), 
                                maxLevel = 2, 
                                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
colors = np.random.randint(0,255, (100, 3))

ret, frame = cap.read()

frame_gray_init = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

edges = cv.goodFeaturesToTrack(frame_gray_init, mask = None, **parameters_shitomasi)
#print(edges)
#print(len(edges))

mask = np.zeros_like(frame)
#print(mask)
#print(np.shape(mask))

while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    new_edges, status, erros = cv.calcOpticalFlowPyrLK(frame_gray_init, 
                                                        frame_gray, 
                                                        edges, 
                                                        None, 
                                                        **parameters_lucas_kanade)
    news = new_edges[status == 1]
    olds = edges[status == 1]

    for i, (new, old) in enumerate(zip(news, olds)):
        a, b = new.ravel()
        c, d = old.ravel()

        mask = cv.line(mask, (a, b), (c, d), colors[i].tolist(), 2)

        frame = cv.circle(frame, (a,b), 5, colors[i].tolist(), -1)

    img = cv.add(frame, mask)

    cv.imshow('Optical Flow', img)
    if cv.waitKey(1) == 13:
        break

    frame_gray_init = frame_gray.copy()
    edges = news.reshape(-1,1,2)

cv.destroyAllWindows()
cap.release()