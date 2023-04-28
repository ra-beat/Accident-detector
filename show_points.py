import cv2 as cv
import numpy as np
import json
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

traffic_json = os.getenv("TRAFFIC_JSON")  # TRAFFIC_JSON PARKING_JSON
parking_json = os.getenv("PARKING_JSON")

traffic = json.load(open(traffic_json))
parking = json.load(open(parking_json))

width, height = 1920, 1080
img = np.zeros((height, width, 3), dtype=np.uint8)


def get_color(weigh):
    if weigh <= 3:
        return (255, 0, 0)
    if weigh <= 6:
        return (255, 255, 0)
    if weigh <= 20:
        return (0, 255, 0)


setT = set(traffic)
setP = set(parking)

res = setP & setT
res = res & setP

print(res)
# for key in traffic.keys():
#     weigh = int(traffic[key])
#     coor = map(int, eval(key))
#     cv.circle(img, tuple(coor), weigh, (0, 255, 0), -1)
#
# for key in parking.keys():
#     weigh = int(parking[key])
#     coor = map(int, eval(key))
#     cv.circle(img, tuple(coor), weigh,(255, 0, 0), -1)
#
#
for key in res:
    coor = map(int, eval(key))
    cv.circle(img, tuple(coor), 10, (255, 255, 255), -1)


cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
