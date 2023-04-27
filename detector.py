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

print(stream)

cap = cv.VideoCapture(stream, cv.CAP_FFMPEG)
model = YOLO("yolov8x.pt")
interval_video = 4
stats = {}
crash = {}
crash_json = {}

parking_json = {}
neighbour_parking = {}

traffic_json = {}
neighbour_traffic = {}

all_stat_json = {}

threshold_coordinates = 2  # Погрешность по координатам
threshold_reiteration = 5  # Пороговое значение повторений, возможно использовать для определения соседей

frame_count = 0
last_time = 10


def detector_cars(frame):
    results = model(frame, conf=0.22)  # Распознавание машины.
    results = results[0].numpy()
    cars_results = results.boxes[results.boxes.cls == 2].xywh[:, :2]  # Координаты, для кругов
    return cars_results


def statistics(car):
    for key in stats.keys():
        all_stat_json[str(key)] = int(stats[key])

        if fault_coordinates(key, car):
            stats[key] += 1
            crash[key] = stats[key]

            if reiteration_coordinates(stats[key]):
                crash_json[str(key)] = int(stats[key])
                crash[key] = stats[key]

        if neighbour_parking_detect(key, car):
            parking_json[str(key)] = int(stats[key])
            neighbour_parking[key] = stats[key]

        if neighbour_traffic_detect(key, car):
            traffic_json[str(key)] = int(stats[key])
            neighbour_traffic[key] = stats[key]

    print("Парковка => ", len(neighbour_parking))
    print("Проезжая часть => ", len(neighbour_traffic))
    print("Всего =>", len(stats))
    key = tuple(car)
    stats[key] = 1
    return crash


def crash_show(crashs, frame):
    if len(crashs) != 0:
        for crash in crashs:

            xy = tuple(int(x) for x in crash)
            option = crashs[crash]
            rgb = marker_rgb(option)

            cv.circle(frame, xy, option, rgb)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        cv.imshow('Frame', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()


def marker_rgb(option):
    steps = option
    values = np.linspace(0, 1, steps)

    r = int(round(values[option - 3] * 255))
    g = int(round((1 - values[option - 3]) * values[option - 3] * 2 * 255))
    b = int(round((1 - values[option - 3]) * 255))
    color_tuple = (r, g, b)
    return color_tuple


def stats_write(stats_list):
    print('Write Stats')
    with open(stats_json, 'w') as file:
        json_stat = json.dumps(stats_list)
        file.write(json_stat)
        return True


def neighbour_parking_write(neighbour_list):
    print("Write neighbour parking")
    with open(neighbour_parking_json, 'w') as file:
        json_neighbour = json.dumps(neighbour_list)
        file.write(json_neighbour)
        return True


def neighbour_traffic_write(neighbour_list):
    print("Write neighbour traffic")
    with open(neighbour_traffic_json, 'w') as file:
        json_neighbour = json.dumps(neighbour_list)
        file.write(json_neighbour)
        return True


def fault_coordinates(key, car):
    if np.abs(np.array(key) - car).max() <= threshold_coordinates:
        return True
    else:
        return False


def reiteration_coordinates(key):
    if key > threshold_reiteration:
        return True
    else:
        return False


def neighbour_parking_detect(key, car):
    if stats[key] > threshold_reiteration and np.abs(np.array(key) - car).max() <= stats[key]:
        print("*")
        return True
    else:
        return False


def neighbour_traffic_detect(key, car):
    if stats[key] < threshold_reiteration and np.abs(np.array(key) - car).max() >= stats[key]:
        print("+")
        return True
    else:
        return False


if not cap.isOpened():
    print("Ошибка открытия файла или потока")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if (cv.getTickCount() - last_time) / cv.getTickFrequency() >= interval_video:
            cars_results = detector_cars(frame)

            for cars_result in cars_results:
                crash = statistics(cars_result)
                crash_show(crash, frame)

                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
            cv.imshow('Frame', frame)
            last_time = cv.getTickCount()

        frame_count += 1
        if cv.waitKey(25) & 0xFF == ord('q'):
            if stats_write(all_stat_json):
                if neighbour_parking_write(parking_json):
                    if neighbour_parking_write(traffic_json):
                        break

    else:
        break

cap.release()
cv.destroyAllWindows()
