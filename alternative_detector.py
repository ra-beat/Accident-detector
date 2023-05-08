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

stats = {}
parking = {}
traffic = {}

threshold_coordinates = 10  # Погрешность по координатам
threshold_parking = 6  # Пороговое значение повторений
threshold_max = 10  # Максимальное возможное количество повторений
threshold_min = -1
max_reiteration = 15  # Максимальное количество значений повторений

count = 0
factor = 0.8
iterator = threshold_parking

# def statistics(car_box, frame):
#     global stats, traffic, parking, iterator
#     car_box = tuple(car_box)
#
#     if not check_key(car_box):
#         stats[car_box] = 0
#     else:
#         key = get_key(car_box)
#         if stats[key] < threshold_max:
#             stats[key] += 1
#
#     if iterator == 4:
#         stats = {key:val for key,val in stats.items() if val == 0}
#         iterator = 0
#     iterator += 1
#
#     for key in stats.keys():
#         if stats[key] < threshold_parking:
#             traffic[key] = stats[key]
#
#         else:
#             parking[key] = stats[key]
#
#
#     show(frame)
#     # split()

def statistics(car_box, frame):
    global stats

    filter = {}
    factor = 0.8

    for key in stats.keys():
        if bbox_iou(car_box, key) > factor:
            if stats[key] < max_reiteration:
                stats[key] += 1
        else:
            filter[tuple(car_box)] = stats[key]

        if bbox_iou(car_box, key) > factor and stats[key] >= threshold_parking:
            parking[key] = stats[key]

        elif threshold_parking > stats[key] > 3:
            traffic[key] = stats[key]

    for key in filter.keys():
        stats.pop(key, None)

    stats[tuple(car_box)] = 0
    # show(frame)


def check_key(car_box):
    if len(stats) > 0:
        for key in stats.keys():
            if bbox_iou(key, car_box) > 0.7:
                return True
            else:
                return False
    return False

def get_key(car_box):
    if len(stats) > 0:
        for key in stats.keys():
            if bbox_iou(key, car_box) > 0.7:
                return key


# def show(frame):
#     width, height = 1920, 1080
#     img = np.zeros((height, width, 3), dtype=np.uint8)
#
#     for key in parking.keys():
#         xy = tuple(map(int, key[:2]))
#         cv.circle(frame, xy, 10, (0, 255, 0), -1)
#
#     for key in traffic.keys():
#         xy = tuple(map(int, key[:2]))
#         cv.circle(frame, xy, 4, (0, 0, 255), -1)
#
#     cv.namedWindow('Frame', cv.WINDOW_NORMAL)
#     cv.imshow('Frame', frame)
#     if cv.waitKey(25) & 0xFF == ord('q'):
#         cv.destroyAllWindows()
#         return True
# def stat_filter(stats, threshold_parking):
#     for key in stats.keys():
#         if stats[key] > threshold_parking:
#             yield (key, stats[key])


# def process_traffic(key):
#     res_temp_set = set()
#     res_park_set = set()
#     for park in parking:
#         if bbox_iou(park, key) > 0.7:
#             res_temp_set.add(key)
#         else:
#             res_park_set.add(key)
#     return (res_temp_set, res_park_set)
# def split():
#     global count
#     filter_stats = {}
#     filter_parking = {}
#     for key, value in stats.items():  # проходим по всем парам ключ-значение
#         if value < threshold_parking:  # если значение равно максимальному
#             filter_stats[key] = value  # добавляем в новый словарь
#         elif value == threshold_parking:
#             filter_parking[key] = value
#
#     # if count > threshold_parking:
#     if count == 42:
#         print("-----------------------------------------------------------------------------")
#         print("traffic")
#         print("-----------------------------------------------------------------------------")
#         print(traffic)
#         print("-----------------------------------------------------------------------------")
#         print("filter_parking")
#         print("-----------------------------------------------------------------------------")
#         print(filter_parking)
#
#         filter_traffic = set(filter_stats) | set(parking)
#         res_temp = set()
#         res_park = set()
#         print("*******************************************************************************")
#         print("*******************************************************************************")
#         print("Filter stat", len(filter_stats))
#         print("Parking", len(parking))
#         print(len(res_park))
#         print(len(res_temp))
#         print("*******************************************************************************")
#         print("*******************************************************************************")
#
#         for key in filter_traffic:
#             for park in parking:
#                 if bbox_iou(park, key) > 0.7:
#                     res_temp.add(key)
#                 else:
#                     res_park.add(key)
#
#         res = res_park - res_temp
#
#         # if len(res) > 0:
#         #     print("+++++++++++++++++++++++++ НАЙДЕН итерация: ", count, "+++++++++++++++++++++++++")
#         #     print(filter_traffic)
#         #     time.sleep(600)
#         # time.sleep(600)


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

def load_images_from_folder(folder):
    images = []

    for name in range(0,len(os.listdir(folder))):
        filename = str(name) + ".jpg"
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def detector_cars(frame):
    global count
    results = model(frame, conf=0.4)  # Распознавание машины.
    results = results[0].numpy()
    box = results.boxes[results.boxes.cls == 2].xywh[:, :4]  # размеры объектов
    # split()
    count += 1
    print("------------------------ Итерация ", count, " ------------------------")
    return tuple(box.tolist())  #


start_time = time.time()


def make_images():
    images = load_images_from_folder("test/")
    for frame in images:
        iteration_start_time = time.time()

        if frame is not None:
            cars_boxes = detector_cars(frame)
            for car_box in cars_boxes:
                statistics(car_box, frame)

            print("Парковка => ", len(parking))
            print("Проезжая часть => ", len(traffic))
            print("Всего =>", len(stats))

        if 0xFF == ord('q'):
            break
        iteration_end_time = time.time()
        iteration_time = iteration_end_time - iteration_start_time
        print("Время выполнения итерации: ", iteration_time)


make_images()