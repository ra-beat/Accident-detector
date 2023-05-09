from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import os
import time
from dotenv import load_dotenv
import math
from multiprocessing import Pool

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


model = YOLO("yolov8x.pt")
count = 0
buffer_size = 2
buffer_index = 0
tolerance = 0.5 # погрешность

stats_buffer = np.empty((buffer_size,), dtype=np.ndarray)
parking_buffer = {}

def find_parking():
    global count, parking_buffer
    stats_array = np.array(stats_buffer)
    stats_array = stats_array[:-1]  # Все элементы кроме последнего
    last_stat = stats_array[-1]  # последний
    crashes = []

    for stat in stats_array:
        for stat_row in stat:  # первый
            if len(parking_buffer) == 0:
                parking_buffer[tuple(stat_row)] = 0
            else:
                for last_row in last_stat:  # второй
                    iou = bbox_iou(stat_row, last_row)
                    if iou > 0.8:
                        stat_row = tuple(stat_row)
                        if stat_row not in parking_buffer:
                            parking_buffer[stat_row] = 0
                        else:
                            parking_buffer[stat_row] += 1
                    else:
                        stat_row = tuple(stat_row)
                        if stat_row in parking_buffer:
                            parking_buffer[stat_row] -= 1

    for key in list(parking_buffer.keys()):  # используем list() для создания копии ключей словаря, чтобы избежать RuntimeError
        if parking_buffer[key] <= 0:
            del parking_buffer[key]

    if len(parking_buffer):
        parking_buffer = {k: v for k, v in parking_buffer.items() if v is not None and v < 0}
        print(parking_buffer)

        time.sleep(600)
        mean_value = sum(parking_buffer.values()) / len(parking_buffer)
        max_value = max(parking_buffer.values())
        print("----------******************")
        print("Mean", mean_value)
        print("----------******************")
        traffic = {k: v for k, v in parking_buffer.items() if v is not None and 0 < v < buffer_size}
        parking = {k: v for k, v in parking_buffer.items() if v is not None and v >= buffer_size}
        seven = {k: v for k, v in parking_buffer.items() if v is not None and v == buffer_size}

        # traffic = set(parking) - set(traffic)

        # crash = find_nearest_point(traffic, set(parking.keys()))
        parking = find_nearest_point(set(parking.keys()), traffic)
        crash = set(traffic) - set(parking)
        crashes.append(crash)

        if len(crash) > 0:
            # if set(parking.keys()) < set(crash):
                print("=================== Найдено! =================== ")
                print(crash)
                show(crash, parking, traffic, seven)
                print(parking)
                print("***************************************************")
                print(traffic)
                cv.waitKey(0)





def show(crash, parking, traffic, seven):
    width, height = 1920, 1080
    img = np.zeros((height, width, 3), dtype=np.uint8)
    #
    # for key in crash:
    #     cv.circle(img, tuple(map(int, key[:2])), 20, (255, 0, 0), -1)   # синий
    #
    # for key in traffic:
    #     cv.circle(img, tuple(map(int, key[:2])), 10, (0, 255, 0), -1) # зеленый

    for key in parking:
        cv.circle(img, tuple(map(int, key[:2])), 10, (255, 255, 255), -1) # белый
    #
    # for key in seven:
    #     cv.circle(img, tuple(map(int, key[:2])), 10, (0, 0, 255), -1) # красный
    cv.namedWindow('Frame', cv.WINDOW_NORMAL)
    cv.imshow('Frame', img)

def statistics(car_boxes):
    global stats_buffer

    if stats_buffer is None:
        for i in range(buffer_size-1):
            stats_buffer[i] = car_boxes

    # перезаписываем последние 7 значений в буфере
    for i in range(buffer_size-1):
        stats_buffer[i] = stats_buffer[i+1]
    stats_buffer[buffer_size-1] = car_boxes
    if count >= buffer_size:
        find_parking()

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

def find_nearest_point(set1, set2):
    nearest_point = None
    min_distance = float('inf')
    for point1 in set1:
        for point2 in set2:
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point1
    return [nearest_point]

def load_images_from_folder(folder):
    images = []

    for name in range(0,len(os.listdir(folder))):
        filename = str(name) + ".jpg"
        print("------------------------------------------")
        print("----------------", filename,"--------------------------")
        print("------------------------------------------")

        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def detector_cars(frame):
    global count
    results = model(frame, conf=0.22)  # Распознавание машины.
    results = results[0].numpy()
    box = results.boxes[results.boxes.cls == 2].xywh[:, :4]  # размеры объектов
    count += 1
    print("------------------------ Итерация ", count, " ------------------------")
    print("----------------------------------------------------------------------")
    # if count == 43:
    #     print("Буфер парковки ", parking_buffer)
    #     time.sleep(600)
    return box  #


start_time = time.time()

def make_images():
    images = load_images_from_folder("test/")
    for frame in images:
        iteration_start_time = time.time()

        if frame is not None:
            cars_boxes = detector_cars(frame)
            statistics(cars_boxes)

        if 0xFF == ord('q'):
            break
        iteration_end_time = time.time()
        iteration_time = iteration_end_time - iteration_start_time
        print("Время выполнения итерации: ", iteration_time)


make_images()