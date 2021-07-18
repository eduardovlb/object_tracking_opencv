import cv2 as cv
import sys
from random import randint

# Apenas com opencv 3.4.4

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    if trackerType == tracker_types[0]:
        tracker = cv.legacy_TrackerBoosting
    elif trackerType == tracker_types[1]:
        tracker = cv.legacy_TrackerMIL
    elif trackerType == tracker_types[2]:
        tracker = cv.legacy_TrackerKCF
    elif trackerType == tracker_types[3]:
        tracker = cv.legacy_TrackerTLD
    elif trackerType == tracker_types[4]:
        tracker = cv.legacy_TrackerMedianFlow
    elif trackerType == tracker_types[5]:
        tracker = cv.legacy_TrackerMOSSE
    elif trackerType == tracker_types[6]:
        tracker = cv.legacy_TrackerCSRT
    else:
        tracker = None
        print("Nome incorreto")
        print("Os rastreadores disponível são: ")
        for t in tracker_types:
            print(t)
    
    return tracker


cap = cv.VideoCapture("videos/race.mp4")

ok, frame = cap.read()
if not ok:
    print("Não é possível ler o arquivo de vídeo")
    sys.exit()

bboxes = []
colors = []

while True:
    bbox = cv.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0,255), randint(0, 255)))
    print("Pressione Q para sair das caixas de seleção e começar a rastrear")
    print("Pressione qualquer outra tecla para selecionar o próximo objeto")
    k = cv.waitKey(0) & 0XFF
    if(k == 113):
        break
    
print(f'Caixa delimitadoras selecionadas {bboxes}')
print(f'Cores: {colors}')

trackertype = "CSRT"
multiTracker = cv.MultiTracker_create()

for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackertype), frame, bbox)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    ok, boxes = multiTracker.update(frame)

    for i, newbox in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in newbox]
        cv.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2, 1)

    cv.imshow('MultiTracker', frame)

    if cv.waitKey(1) & 0XFF == 27:
        break



