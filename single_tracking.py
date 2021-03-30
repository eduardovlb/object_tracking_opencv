import cv2
import sys
from random import randint

# Definindo o rastreador a ser utilizado
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[1]

# Criando os rastreadores
if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()


# Abrindo o arquivo do vídeo
video = cv2.VideoCapture('videos/race.mp4')
if not video.isOpened():
    print('Não foi possível carregar o vídeo')
    sys.exit()

# Lendo o vídeo
ok, frame = video.read()
if not ok:
    print('Não foi possível ler o arquivo de vídeo')
    sys.exit()
#print(ok)


# Selecionando objeto para o rastreamento
bbox = cv2.selectROI(frame, False)
#print(bbox)

ok = tracker.init(frame, bbox)
#print(ok)

colors = (randint(0, 255), randint(0, 255), randint(0, 255))
#print(colors)

while True:
    ok, frame = video.read()
    if not ok:
        break

    timer = cv2.getTickCount()
    ok, bbox = tracker.update(frame)
    print(ok, bbox)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ok :
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        cv2.putText(frame, 'Falha no Rastreamento', (100, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.putText(frame, 'FPS' + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)

    cv2.putText(frame, tracker_type + ' Tracker', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break