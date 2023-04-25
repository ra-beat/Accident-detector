import cv2 as cv
import numpy as np
import json
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

stats_json = os.getenv("STATS_JSON")

datas = json.load(open(stats_json))
width, height = 1280, 720
img = np.zeros((height, width, 3), dtype=np.uint8)


def get_color(weigh):
    if weigh <= 3:
        return (255, 0, 0)
    if weigh <= 6:
        return (255, 255, 0)
    if weigh <= 20:
        return (0, 255, 0)


for key in datas.keys():
    weigh = int(datas[key])
    color = get_color(weigh)
    print(type(color))
    coor = map(int, eval(key))
    cv.circle(img, tuple(coor), weigh, color, -1)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
