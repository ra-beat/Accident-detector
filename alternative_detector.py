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


def statistic(results):
    global count
    for result in results:
        car = tuple(map(int, result))

        if stats.get(car) is None:  # проверяет есть ли ключ в массиве, если нет возвращает None
            stats[car] = 1

        if stats[car] != limit_stats:  # если значение не равно лимиту прибавляет 1
            stats[car] += 1

        if stats[car] == limit_stats:
            print(car)



    count += 1
    print("Count=", count)


def detector_car(frame):
    results = model(frame, conf=0.22)
    results = results[0].numpy()
    results = results.boxes[results.boxes.cls == 2].xywh[:, :2]
    statistic(results)


if not cap.isOpened():
    print("Ошибка открытия файла или потока")

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        detector_car(frame)
        last_time = cv.getTickCount()
    else:
        break
