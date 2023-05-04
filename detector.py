from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import os
import time
from dotenv import load_dotenv
import math
from collections import OrderedDict

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

stats = {}
crash = {}
parking = {}
traffic = {}
interval_video = 4

threshold_coordinates = 10  # Погрешность по координатам
threshold_reiteration = 50  # Пороговое значение повторений, возможно использовать для определения соседей
threshold_max = 10  # Максимальное возможное количество повторений


def detector_cars(frame):
    results = model(frame, conf=0.22)  # Распознавание машины.
    results = results[0].numpy()
    box = results.boxes[results.boxes.cls == 2].xywh[:, :4]  # размеры объектов
    return tuple(box.tolist())  #




def statistics(car_box, frame):
    # global stats

    factor = 0.7

    for key in stats.keys():
        if bbox_iou(car_box, key) > factor:
            stats[key] += 1

        # elif stats[key] > 5:
        #     stats[key] -= 1

        if bbox_iou(car_box, key) > factor and stats[key] >= threshold_reiteration:
            parking[key] = stats[key]

        if bbox_iou(car_box, key) > factor and threshold_reiteration > stats[key] > 4:
            traffic[key] = stats[key]


    # stats = {key: stats[key] for key in stats if stats[key] < 0}
    # print("Парковка => ", len(parking))
    # print("Проезжая часть => ", len(traffic))
    # print("Всего =>", len(stats))
    # print("---------------------------",stats)
    stats[tuple(car_box)] = 1
    show(frame)


def accident_show(xy):
    width, height = 1920, 1080
    img = np.zeros((height, width, 3), dtype=np.uint8)
    print("--------------------------")

    cv.circle(img, xy, 20, (255, 255, 255), -1)
    cv.imshow('Frame', img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()


def show(frame):
    width, height = 1920, 1080
    img = np.zeros((height, width, 3), dtype=np.uint8)
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # for key in stats.keys():
    #     print(key, '=>', stats[key])

    for key in parking.keys():
        xy = tuple(map(int, key[:2]))
        cv.circle(gray_image, xy, 10, (0, 255, 0), -1)


    for key in traffic.keys():
        xy = tuple(map(int, key[:2]))
        cv.circle(gray_image, xy, 4, (0, 0, 255), -1)

    cv.namedWindow('Frame', cv.WINDOW_NORMAL)
    cv.imshow('Frame', gray_image)
    if cv.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        return True


def split():
    traf = set(traffic)
    prak = set(parking)


def marker_rgb(option):
    if option <= threshold_reiteration:
        return (0, 255, 0)  # зеленый когда меньше или равен
    if option > threshold_reiteration:
        return (255, 0, 0)  # Красный когда вес больше порогового значения


def bbox_iou(boxA, boxB, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = torch.FloatTensor(boxA)
    box2 = torch.FloatTensor(boxB)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                        b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return float(iou)  # IoU


if not cap.isOpened():
    print("Ошибка открытия файла или потока")

while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        cars_boxes = detector_cars(frame)
        for car_box in cars_boxes:
            statistics(car_box, frame)

            if 0xFF == ord('q'):
                break

    if 0xFF == ord('q'):
        break




