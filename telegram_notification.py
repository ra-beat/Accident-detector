# import telebot
# from datetime import datetime
# import numpy as np
# import cv2
# from PIL import Image as im
# from images_storage import SaveImage
# from telebot.async_telebot import AsyncTeleBot
# import asyncio
#
# save_image = SaveImage("/home/user/Accident-detector/", "road_accident", "test_image", "positive_image", [1, 2, 3],
#                        1, [1, 2, 3], 1280)
#
#
# class TelegramNotification:
#     def __init__(self, token, group_id):
#         self.bot = AsyncTeleBot(token)
#         self.group_id = group_id
#
#         @self.bot.callback_query_handler(func=lambda call: call.data == 'save_image')
#         async def handle_callback(call):
#             img =  cv2.imread(call.message.photo[-1].file_id)
#             save_image.save_positive_image(img)
#
#     async def send_image(self, image, points, caption):
#         for point in points:
#             image = draw_polylines(image, point)
#
#         keyboard = telebot.types.InlineKeyboardMarkup()
#         button = telebot.types.InlineKeyboardButton(text='Авария', callback_data='save_image')
#         keyboard.add(button)
#
#         image = im.fromarray(image)
#         caption = str(caption) + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M")
#         await self.bot.send_photo(self.group_id, photo=image, caption=caption, reply_markup=keyboard)
#
#
#         @self.bot.callback_query_handler(func=lambda call: call.data == 'save_image')
#         def handle_callback(call):
#             SaveImage.save_positive_image(image)
#
#
#
#
# def draw_polylines(image, point):
#     coordinates = []
#     # рассчёт координаты для разрешения изображения
#     shape = image.shape[:2]
#     for i, cord in enumerate(np.reshape(point, -1)):
#
#         if i % 2:
#             coordinates.append(int(round(shape[0] * cord)))
#         else:
#             coordinates.append(int(round(shape[1] * cord)))
#
#     # xyxy в многоугольник, если объект прямоугольник
#     if len(coordinates) == 4:
#         coordinates = [coordinates[0], coordinates[1], coordinates[0], coordinates[3], coordinates[2],
#                        coordinates[3], coordinates[2], coordinates[1], coordinates[0], coordinates[1]]
#     # измяю форму
#     coordinates = np.reshape(np.array(coordinates), (-1, 1, 2))
#     # рисует границы на изображении
#     return cv2.polylines(image, [coordinates], False, (0, 0, 250), thickness=1)

import telebot
from datetime import datetime
import numpy as np
import cv2
from PIL import Image as im
from images_storage import SaveImage
from telebot.async_telebot import AsyncTeleBot
import asyncio

save_image = SaveImage("/home/user/Accident-detector/", "road_accident", "test_image", "positive_image", [1, 2, 3], 1,
                       [1, 2, 3], 1280)


class TelegramNotification:
    def __init__(self, token, group_id):
        self.bot = AsyncTeleBot(token)
        self.group_id = group_id

        @self.bot.callback_query_handler(func=lambda call: call.data == 'save_image')
        async def handle_callback(call):
            img = cv2.imread(call.message.photo[-1].file_id)
            save_image.save_positive_image(img)

    async def send_image(self, image, points, caption):
        for point in points:
            image = draw_polylines(image, point)

        keyboard = telebot.types.InlineKeyboardMarkup()
        button = telebot.types.InlineKeyboardButton(text='Авария', callback_data='save_image')
        keyboard.add(button)

        image = im.fromarray(image)
        caption = str(caption) + ' ' + datetime.now().strftime("%Y-%m-%d %H:%M")
        await self.bot.send_photo(self.group_id, photo=image, caption=caption, reply_markup=keyboard)

    async def close(self):
        await self.bot.close()


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
        coordinates = [coordinates[0], coordinates[1], coordinates[0], coordinates[3], coordinates[2], coordinates[3],
                       coordinates[2], coordinates[1], coordinates[0], coordinates[1]]

    # измяю форму
    coordinates = np.reshape(np.array(coordinates), (-1, 1, 2))
    # рисует границы на изображении
    return cv2.polylines(image, [coordinates], False, (0, 0, 250), thickness=1)
