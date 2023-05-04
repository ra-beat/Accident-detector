from ultralytics import YOLO
import cv2 as cv
import numpy as np
import json
import os
from dotenv import load_dotenv

# dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
# if os.path.exists(dotenv_path):
#     load_dotenv(dotenv_path)
#
# stream = os.getenv("STREAM")
# stats_json = os.getenv("STATS_JSON")
# neighbour_parking_json = os.getenv("PARKING_JSON")
# neighbour_traffic_json = os.getenv("TRAFFIC_JSON")
#
# # cap = cv.VideoCapture(stream, cv.CAP_FFMPEG)
# cap = cv.VideoCapture('crash.mp4')
# cap.set(cv.CAP_PROP_POS_MSEC, 10000)  # берется кадр каждые 10 секунд
# model = YOLO("yolov8x.pt")
#
# interval_video = 1
# last_time = 0
#
# limit_stats = 15  # лимит значений повторений для авто
# len_stats = 100  # ограничение
# stats = {}
# count = 0
import torch
import math
crash = {(1681.5, 744.0, 107.0, 58.0), (1815.5, 734.5, 103.0, 57.0), (1681.0, 743.5, 108.0, 55.0), (692.0, 1044.0, 118.0, 70.0), (1903.5, 1056.0, 31.0, 46.0), (692.5, 1045.0, 117.0, 68.0), (692.0, 1044.0, 120.0, 70.0), (691.5, 1044.5, 115.0, 69.0), (692.5, 1045.0, 119.0, 68.0), (1401.0, 802.5, 124.0, 69.0), (1401.0, 803.0, 122.0, 68.0), (904.0, 964.0, 124.0, 80.0), (1904.5, 1057.5, 29.0, 43.0), (904.0, 964.0, 126.0, 80.0), (1682.5, 743.5, 105.0, 57.0), (1905.0, 1057.5, 28.0, 43.0), (1682.0, 743.5, 106.0, 59.0), (903.0, 963.5, 126.0, 81.0), (1905.0, 1056.0, 28.0, 46.0), (691.0, 1044.0, 116.0, 70.0), (903.5, 964.0, 125.0, 80.0), (1815.5, 734.0, 103.0, 58.0), (1400.5, 802.5, 121.0, 69.0), (1680.5, 743.5, 109.0, 55.0), (1681.5, 743.0, 105.0, 56.0), (1906.0, 725.0, 26.0, 38.0), (1904.0, 1057.0, 30.0, 44.0), (1680.5, 742.5, 107.0, 55.0), (1361.5, 1036.5, 143.0, 81.0), (1400.0, 803.0, 122.0, 68.0), (903.0, 964.0, 124.0, 80.0), (692.5, 1044.0, 119.0, 70.0), (1362.0, 1035.5, 144.0, 83.0), (1906.0, 724.0, 26.0, 38.0), (904.0, 962.5, 126.0, 83.0), (1815.0, 734.0, 104.0, 58.0), (1815.0, 734.5, 104.0, 59.0), (1904.5, 1056.5, 29.0, 45.0), (1815.5, 735.0, 105.0, 58.0), (1400.5, 802.5, 123.0, 69.0), (904.0, 964.0, 124.0, 82.0), (903.5, 963.5, 123.0, 81.0), (1360.5, 1036.0, 143.0, 82.0), (903.5, 963.5, 125.0, 81.0), (904.0, 963.5, 124.0, 81.0), (1906.0, 726.0, 26.0, 36.0), (1906.0, 725.5, 26.0, 37.0), (1681.0, 744.0, 108.0, 58.0), (1400.0, 802.5, 120.0, 69.0), (1904.0, 1056.0, 30.0, 46.0), (692.5, 1045.5, 119.0, 67.0), (1904.5, 1057.0, 29.0, 44.0), (1814.5, 734.5, 103.0, 57.0), (1680.0, 744.0, 108.0, 56.0), (691.5, 1045.5, 117.0, 67.0), (1680.0, 744.0, 110.0, 56.0), (1903.5, 1056.5, 31.0, 45.0), (1905.5, 724.5, 27.0, 39.0), (692.0, 1044.5, 120.0, 69.0), (1906.0, 724.0, 26.0, 40.0), (1680.5, 744.0, 107.0, 58.0), (692.0, 1045.0, 118.0, 68.0), (1815.5, 735.0, 103.0, 58.0), (1399.0, 803.0, 120.0, 68.0), (693.0, 1044.0, 118.0, 70.0), (1361.0, 1035.0, 144.0, 84.0), (1815.0, 735.0, 104.0, 60.0), (1361.5, 1035.5, 143.0, 83.0), (1361.5, 1035.0, 143.0, 84.0), (1681.5, 743.0, 107.0, 58.0), (1816.0, 734.5, 104.0, 57.0), (1401.0, 802.5, 122.0, 69.0), (903.5, 963.5, 125.0, 83.0)}
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



print(bbox_iou((1681.5, 744.0, 107.0, 58.0), (1681.0, 743.5, 108.0, 55.0)))
#
# if not cap.isOpened():
#     print("Ошибка открытия файла или потока")
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if ret:
#         for key in crash:
#             xy = tuple(map(int, key[:2]))
#             cv.circle(frame, xy, 20, (0, 0, 255), -1)
#             last_time = cv.getTickCount()
#             cv.namedWindow('Frame', cv.WINDOW_NORMAL)
#             cv.imshow('Frame', frame)
#             if cv.waitKey(25) & 0xFF == ord('q'):
#                 cap.release()
#                 cv.destroyAllWindows()
#     else:
#         break
