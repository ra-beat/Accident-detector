import telebot
from datetime import datetime
import numpy as np
import cv2
from PIL import Image as im


class TelegramNotification:
    def __init__(self, token, group_id):
        self.bot = telebot.TeleBot(token)
        self.group_id = group_id

    def send_image(self, image, points, caption):
        for point in points:
            image = draw_polylines(image, point)

        image = im.fromarray(image)
        caption = caption + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M")
        self.bot.send_photo(self.group_id, photo=image, caption=caption)


def draw_polylines(image, point):
    coordinates = []
    # рассчёт координаты для разрешения изображения
    shape = image.shape[:2]
    for i, cord in enumerate(np.reshape(point, -1)):

        if i % 2:
            coordinates.append(int(round(shape[0] * cord)))
        else:
            coordinates.append(int(round(shape[1] * cord)))

    # xyxy в многоугольник, если объект прямоугольник
    if len(coordinates) == 4:
        coordinates = [coordinates[0], coordinates[1], coordinates[0], coordinates[3], coordinates[2],
                       coordinates[3], coordinates[2], coordinates[1], coordinates[0], coordinates[1]]
    # измяю форму
    coordinates = np.reshape(np.array(coordinates), (-1, 1, 2))
    # рисует границы на изображении
    return cv2.polylines(image, [coordinates], False, (0, 0, 250), thickness=1)
