from ultralytics import YOLO
import cv2 as cv
import numpy as np
import json
import os
import time
from dotenv import load_dotenv
from collections import OrderedDict

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

stream = os.getenv("STREAM")
stats_json = os.getenv("STATS_JSON")
neighbour_parking_json = os.getenv("PARKING_JSON")
neighbour_traffic_json = os.getenv("TRAFFIC_JSON")

cap = cv.VideoCapture(stream, cv.CAP_FFMPEG)
cap.set(cv.CAP_PROP_POS_MSEC, 10000)  # берется кадр каждые 10 секунд
model = YOLO("yolov8x.pt")

stats = {}
crash = {}
parking = {}
traffic = {}
interval_video = 4

threshold_coordinates = 14  # Погрешность по координатам
threshold_reiteration = 6  # Пороговое значение повторений, возможно использовать для определения соседей
threshold_max = 10 # Максимальное возможное количество повторений


def detector_cars(frame):
    results = model(frame, conf=0.22)  # Распознавание машины.
    results = results[0].numpy()
    results = results.boxes[results.boxes.cls == 2].xywh[:, :2]  # Координаты, для кругов
    return results


def statistics(car):
    car = tuple(map(int, car))
    for key in stats.keys():
        if np.abs(np.array(key) - car).max() <= threshold_coordinates:
            if stats[key] <= threshold_max:
                stats[key] += 1

        # Вот здесь не происходит проверка на погрешность от этого появляются "битые данные"
        # Для того что бы значения не дублировались, нужно брать среднее значение и записывать в спписок

        if stats[key] > threshold_reiteration and np.abs(np.array(key) - car).max() <= stats[key]:
            parking[key] = stats[key]

        if stats[key] < threshold_reiteration and np.abs(np.array(key) - car).max() <= threshold_coordinates:
            traffic[key] = stats[key]

    print("Парковка => ", len(parking))
    print("Проезжая часть => ", len(traffic))
    print("Всего =>", len(stats))
    key = tuple(car)
    stats[key] = 1


def accident_show(xy):
    width, height = 1920, 1080
    img = np.zeros((height, width, 3), dtype=np.uint8)
    print("--------------------------")

    cv.circle(img, xy, 20, (255,255,255), -1)
    cv.imshow('Frame', img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()

def show():
    width, height = 1920, 1080
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for par in parking.keys():
        cv.circle(img, par, 10, (255,255,255), -1)
        cv.imshow('Frame', img)

    for key in traffic.keys():
        cv.circle(img, key, 4, (0, 0, 255), -1)
        cv.imshow('Frame', img)

    cv.imshow('Frame', img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        return True




def marker_rgb(option):
    if option <= threshold_reiteration:
        return (0, 255, 0) # зеленый когда меньше или равен
    if option > threshold_reiteration:
        return (255, 0, 0)  # Красный когда вес больше порогового значения


def play():

    if not cap.isOpened():
        print("Ошибка открытия файла или потока")

    while cap.isOpened() and len(stats) < 4000:
        ret, frame = cap.read()
        if ret:
            cars_results = detector_cars(frame)

            for cars_result in cars_results:
                statistics(cars_result)

                if 0xFF == ord('q'):
                    break

        if 0xFF == ord('q'):
            break
    show()
    time.sleep(600)



play()
