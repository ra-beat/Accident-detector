import time
import numpy as np
import os
import os.path
import cv2
from pathlib import Path
from numpy import random


# Сохранение изображений НЕЗАБЫТЬ УБРАТЬ КОМЕНТЫ!
class SaveImage:

    def __init__(self, main_work_directory, path_road_accident, path_test, names_obj,
                 car_cls, image_scale, img_size=640):

        self.image_scale = image_scale
        self.car_cls = car_cls
        self.path_road_accident = str(Path(main_work_directory + path_road_accident).absolute()) + '/'
        self.path_result = str(Path(main_work_directory + path_test).absolute()) + '/'
        self.img_size = img_size
        self.names_obj = names_obj
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names_obj))]
        self.image_quality = 50

    @staticmethod
    def __get_time():
        return time.strftime("%Y%m%d", time.localtime())

    @staticmethod
    def __get_file_list(path):
        # возвращает список файлов в каталоге
        return os.listdir(path)

    # Получение имени файла для сохранения нового изображения
    def __get_name_new_file(self, path):

        current_time = self.__get_time()
        #  поиск доступных файлов
        actual_file = [x.split(".") for x in self.__get_file_list(path) if x.find(current_time) > -1]
        # поиск текущего номера файла
        file_number = 0

        if actual_file:
            for item in actual_file:
                try:
                    file_number = max(int(item[0].replace(current_time, "")), file_number)
                except ValueError:
                    pass
            file_number += 1

        # вернуть имя нового файла
        return current_time + str(file_number).zfill(5)

    # сохраняет изображение по указанному пути, с указанным размером
    def __save_image(self, img, path, size, quality=0):

        shape = img.shape[:2]  # текущий формат [height, width]
        new_shape = (size, size)
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = min(ratio, 1.0)
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))

        # ресайз
        if shape[::-1] != new_unpad and size:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # качество изображения
        if not quality:
            quality = self.image_quality

        # сохранение изображения
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return

    # возвращает путь с каталогом, названным в соответствии с текущей датой
    def __get_new_img_sub_path(self, path):

        str_date = self.__get_time()
        path = path + '/' + str_date + '/'

        if not os.path.exists(path):  # создать новый каталог с именем данных
            os.makedirs(path)

        num_dir = str(len(os.listdir(path)) + 1)
        path = path + num_dir + '/'

        if not os.path.exists(path):  # создать новый подкаталог с номером имени
            os.makedirs(path)

        return path, str_date + '/' + num_dir

    # возвращает путь с каталогом, названным в соответствии с текущей датой
    def __get_new_date_path(self, path):

        str_date = self.__get_time()
        path = path + '/' + str_date + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        return path, str_date + "/"

    # подготавливает полученное изображение об инциденте и сохраняет его в файловой системе
    def save_road_accident_image(self, img, points):

        path_img, link_image = self.__get_new_date_path(self.path_road_accident)
        file_name = self.__get_name_new_file(path_img)

        image = np.copy(img)

        # записывает красные границы объекта
        for point in points:

            image = self.draw_polylines(image, point)
        self.__save_image(image, path_img + file_name + '.jpg', self.img_size)
        return path_img + file_name + '.jpg', file_name

    # вырезает из изображения
    @staticmethod
    def cutout_image(image, point):

        h, w, _ = image.shape
        x1, x2, x3, x4, y1, y2, y3, y4 = point.reshape(-1)
        x1, x2, x3, x4 = x1 * w, x2 * w, x3 * w, x4 * w
        y1, y2, y3, y4 = y1 * h, y2 * h, y3 * h, y4 * h

        src = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype="float32")
        dst = np.array([[0, 0], [144, 0], [0, 48], [144, 48]], dtype="float32")

        m = cv2.getPerspectiveTransform(src, dst)
        out_img = cv2.warpPerspective(image, m, (144, 48))  # 144, 48
        return out_img

    # Применяет разметку с выделенным событием к изображению
    @staticmethod
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

    # сохранение тне обработанного изображения
    def save_test_image(self, stream_id, image):

        cv2.imwrite(self.path_result + str(stream_id) + '_test.jpg', image)
        return

    # Показать результат обнаружения для детектора качества теста
    def show_detection_result(self, det_array, image, stream_id):

        image = image.copy()
        for *xyxy, conf, cls in reversed(det_array):

            label = f'{self.names_obj[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)], line_thickness=1)

        cv2.imwrite(self.path_result + str(stream_id) + '_image.jpg', image)
        return


def plot_one_box(x, img, color=None, label=None, line_thickness=3):

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
