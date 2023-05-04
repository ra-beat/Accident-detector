from ultralytics import YOLO
import cv2 as cv
import numpy as np
import json
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

stream = os.getenv("STREAM")
stats_json = os.getenv("STATS_JSON")
neighbour_parking_json = os.getenv("PARKING_JSON")
neighbour_traffic_json = os.getenv("TRAFFIC_JSON")

# cap = cv.VideoCapture(stream, cv.CAP_FFMPEG)
cap = cv.VideoCapture('crash.mp4')
cap.set(cv.CAP_PROP_POS_MSEC, 10000)  # берется кадр каждые 10 секунд
model = YOLO("yolov8x.pt")

interval_video = 1
last_time = 0

limit_stats = 15  # лимит значений повторений для авто
len_stats = 100  # ограничение
stats = {}
count = 0
crash = {(1362.0, 1035.5, 144.0, 83.0), (1681.5, 744.0, 107.0, 58.0), (1815.0, 734.0, 104.0, 58.0), (691.5, 1045.5, 117.0, 67.0), (1680.5, 744.0, 107.0, 58.0), (1905.5, 724.5, 27.0, 39.0), (1682.0, 743.5, 106.0, 59.0), (1906.0, 724.0, 26.0, 40.0), (693.0, 1044.0, 118.0, 70.0), (1815.0, 735.0, 104.0, 60.0), (1681.5, 743.0, 107.0, 58.0), (692.5, 1044.0, 119.0, 70.0)}



if not cap.isOpened():
    print("Ошибка открытия файла или потока")

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        for key in crash:
            xy = tuple(map(int, key[:2]))
            cv.circle(frame, xy, 20, (0, 0, 255), -1)
            last_time = cv.getTickCount()
            cv.namedWindow('Frame', cv.WINDOW_NORMAL)
            cv.imshow('Frame', frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv.destroyAllWindows()
    else:
        break
