import torch
import math
import numpy as np
import cv2
from datetime import datetime, timedelta


class RoadAccidentFinder:

    def __init__(self, car_cls,
                 min_static_time=80, frame_capture_interval=10,
                 max_point_attenuation=2000, memory_matrix_size=(240, 430), image_resolution=1280):
        # интервал захвата кадра секунд
        self.frame_capture_interval = frame_capture_interval
        # размер матрицы памяти точек объекта
        self.memory_matrix_size = memory_matrix_size
        # номер класса автомобиля
        self.car_cls = car_cls
        # время обнаружения статического объекта
        self.static_object_time = min_static_time / self.frame_capture_interval
        # хранилище для статических объектов
        self.storage_static_objects = {}  # {camera_id: object_array, ...}
        # хранилище матрицы
        self.storage_matrix_memory = {}
        # входные координаты для масштабирования
        self.input_resolution = \
            np.array([image_resolution, image_resolution / 1.777, image_resolution, image_resolution / 1.777, 1, 1])
        # таймер расписания (минуты)
        self.time_interval = 15
        self.time_schedule = {}
        # максимальный шаг затухания точки за одну минуту
        self.max_point_attenuation = max_point_attenuation
        # параметры подключения камеры
        self.cameras_connection_data = []
        self.cameras_id = []  # [camera_id 1, camera_id 2,.., camera_id n]
        # хранилище объектов в реально времени
        self.storage_time_static = {}  # {camera_id: [time_1, time_2, ...time 20], ...}
        self.storage_time_length = 20

        # хранилище количества уникальных объектов
        self.count_unique_object = {}  # {camera_id: 23, ... }

    def set_camera_connection_list(self, cameras_connection_data):
        if self.cameras_connection_data != cameras_connection_data:
            self.cameras_id = [_[0] for _ in cameras_connection_data]  # извлечение ИД камер
            self.cameras_connection_data = cameras_connection_data
            # обновление данных обработки и удаление устаревших данных
            self.storage_matrix_memory = \
                {key: self.storage_matrix_memory[key]
                if key in self.storage_matrix_memory else np.zeros(self.memory_matrix_size, dtype=np.uint16)
                 for key in self.cameras_id}
            self.storage_static_objects = \
                {key: self.storage_static_objects[key] if key in self.storage_static_objects else np.empty((0, 7))
                 for key in self.cameras_id}
            self.time_schedule = \
                {key: self.time_schedule[key]
                if key in self.time_schedule else datetime.now() - timedelta(minutes=self.time_interval)
                 for key in self.cameras_id}
            self.storage_time_static = \
                {key: self.storage_time_static[key]
                if key in self.storage_time_static else [int(self.static_object_time / 2)] * self.storage_time_length
                 for key in self.cameras_id}
            self.count_unique_object = \
                {key: self.count_unique_object[key] if key in self.count_unique_object else 0
                 for key in self.cameras_id}

        return

    # проверка кадра на наличие инцидентов
    def road_accident_process(self, camera_id, prediction_data):
        # обработка массива, отбор только массива объектов
        try:
            object_car = prediction_data[prediction_data[:, -1] == self.car_cls]
        except IndexError:
            return None

        accident_object = None  # Для аварий

        static_object_time = self.get_static_object_time(camera_id)

        # если нет авто в массиве
        if not torch.any(object_car):
            return accident_object

        # преобразование pytorch в массив numpy и абсолютизация координат
        object_car = object_car.detach().numpy()
        object_car = object_car / self.input_resolution
        # отфильтрованный массив автомобилей
        self.storage_static_objects[camera_id], active_object, static_object, time_processing, temporary_static = \
            self.search_active_objects(self.storage_static_objects[camera_id], object_car, static_object_time)
        # сохранение новый авто
        self.save_point_object(camera_id, self.get_median_point(active_object))
        # сохранение количества активных объектов
        self.count_unique_object[camera_id] += len(active_object)
        # сохранение новых статичных объектов
        self.save_lifetime_static(camera_id, temporary_static)
        # проверка планировщика
        if self.check_schedule(camera_id):
            # smooth fading of points on matrix
            self.process_fading_points(camera_id, self.get_point_attenuation(camera_id))
        # проверка новго статичного объекта
        if static_object.any():
            # расчет времени
            obj_time = [(datetime.now() - timedelta(seconds=i * self.frame_capture_interval)).strftime("%H:%M:%S")
                        for i in time_processing]
            for c, object_point in enumerate(static_object):
                if self.check_status_object(camera_id, object_point):
                    accident_object = static_object, obj_time[c]
                else:
                    # удаление статичных объектов из матрицы
                    self.delete_points(camera_id, object_point)

        return accident_object

    # сохраняем время наблюдения за статическим объектом в массив для дальнейшей обработки
    def save_lifetime_static(self, camera_id, lifetime_objects):

        min_num_point = 7
        lifetime = []
        # поиск авто и сохранение
        for static_object in lifetime_objects:
            if static_object[-3] > self.static_object_time / 2:
                object_width, object_height, object_point = self.calculate_size_object(static_object[0:-3])
                # поиск координат
                y1 = int(max(0, object_point[0] - object_height / 2))
                x1 = int(max(0, object_point[1] - object_width / 2))
                y2 = int(min(self.memory_matrix_size[0], object_point[0] + object_height / 2))
                x2 = int(min(self.memory_matrix_size[1], object_point[1] + object_width / 2))
                if np.count_nonzero(self.storage_matrix_memory[camera_id][y1:y2, x1:x2]) >= min_num_point:
                    lifetime.append(static_object[-3])
            else:
                lifetime.append(static_object[-3])

        if lifetime:
            self.storage_time_static[camera_id].append(np.mean(lifetime))
            if len(self.storage_time_static[camera_id]) > self.storage_time_length:
                del self.storage_time_static[camera_id][0]
        return

    # получаем среднее время слежения за статическим объектом
    def get_static_object_time(self, camera_id):
        mean_lifetime = np.mean(self.storage_time_static[camera_id])
        return round(min(
            self.static_object_time * 2,
            self.static_object_time + mean_lifetime * (1 + mean_lifetime / self.static_object_time)))

    # вычисляет размер объекта в пикселях на тепловой карте, а также координаты центра объекта
    def calculate_size_object(self, object_point):

        object_width = int(abs(object_point[0] - object_point[2]) * self.memory_matrix_size[1])
        object_height = int(abs(object_point[1] - object_point[3]) * self.memory_matrix_size[0])
        # объект среднего размера
        object_point = (abs(object_point[1] - (object_point[1] - object_point[3]) / 2) * self.memory_matrix_size[0],
                        abs(object_point[0] - (object_point[0] - object_point[2]) / 2) * self.memory_matrix_size[1])
        return object_width, object_height, object_point

    # проверяет, насколько обнаруженный статический объект влияет на движение других объектов (влияет - Верно, не влияет - Ложь)
    def check_status_object(self, camera_id, object_point):

        # проверка положения объекта
        min_num_point = 13
        # максимальный процент разности луча
        max_differences = 0.5
        # длинна поискового луча
        length_beam = 3
        # расчёт размера объекта на матрице
        object_width, object_height, object_point = self.calculate_size_object(object_point)
        half_size = min(object_width, object_height) / 2  # половина максимального размера объекта
        # координаты сканирующего луча
        beam_coordinates = [
            [max(0, object_point[0] - half_size),  # Y1
             max(0, object_point[1] - half_size * length_beam),  # X1
             min(self.memory_matrix_size[0], object_point[0] + half_size),  # Y2
             min(self.memory_matrix_size[1], object_point[1] + half_size * length_beam)],  # X2
            [min(self.memory_matrix_size[0], object_point[0] + half_size),  # Y3
             max(0, object_point[1] - half_size * length_beam),  # X3
             max(0, object_point[0] - half_size),  # Y4
             min(self.memory_matrix_size[1], object_point[1] + half_size * length_beam)],  # X4
            [max(0, object_point[0] - half_size * length_beam),  # Y5
             max(0, object_point[1] - half_size),  # X5
             min(self.memory_matrix_size[0], object_point[0] + half_size * length_beam),  # Y6
             min(self.memory_matrix_size[1], object_point[1] + half_size)],  # X6
            [max(0, object_point[0] - half_size * length_beam),  # Y7
             min(self.memory_matrix_size[1], object_point[1] + half_size),  # X7
             min(self.memory_matrix_size[0], object_point[0] + half_size * length_beam),  # Y8
             max(0, object_point[1] - half_size)]  # X8
        ]

        # проверка положения объекта
        for beam in beam_coordinates:
            # beam a
            y_a1 = int(min(object_point[0], beam[0]))
            y_a2 = int(max(object_point[0], beam[0]))
            x_a1 = int(min(object_point[1], beam[1]))
            x_a2 = int(max(object_point[1], beam[1]))
            count_a = np.count_nonzero(self.storage_matrix_memory[camera_id][y_a1:y_a2, x_a1:x_a2])
            # beam b
            y_b1 = int(min(object_point[0], beam[2]))
            y_b2 = int(max(object_point[0], beam[2]))
            x_b1 = int(min(object_point[1], beam[3]))
            x_b2 = int(max(object_point[1], beam[3]))
            count_b = np.count_nonzero(self.storage_matrix_memory[camera_id][y_b1:y_b2, x_b1:x_b2])
            # check result
            if count_a < min_num_point or count_b < min_num_point:  # not enough point
                continue
            # calculate ratio beam
            beam_ratio = 1 - min(count_a, count_b) / max(count_a, count_b)

            if beam_ratio < max_differences:  # the ratio is within acceptable limits
                return True
        return False  # accident not found

    # для получения матрицы изображения
    def get_test_map(self, camera_id):
        return np.array(self.storage_matrix_memory[camera_id] / 256, dtype=np.uint8)

    # удаление точек вокрук статического объекта
    def delete_points(self, camera_id, point):
        # вычисление размера обеъекта на матрице
        object_width, object_height, object_point = self.calculate_size_object(point)
        # вычисление координат
        y1 = int(max(0, object_point[0] - object_height / 2))
        x1 = int(max(0, object_point[1] - object_width / 2))
        y2 = int(min(self.memory_matrix_size[0], object_point[0] + object_height / 2))
        x2 = int(min(self.memory_matrix_size[1], object_point[1] + object_width / 2))
        # обнуление среза статических объектов
        self.storage_matrix_memory[camera_id][y1:y2, x1:x2] = 0
        return

    # затухание точек
    def process_fading_points(self, camera_id, point_attenuation):
        """the process of fading points on heat map"""
        # radius for searching for neighboring points
        search_radius = 2
        max_points = (1 + search_radius + search_radius) ** 2
        # delete small item
        self.storage_matrix_memory[camera_id][self.storage_matrix_memory[camera_id] < point_attenuation] = 0
        # search activ point and decrement
        indexes = np.where(self.storage_matrix_memory[camera_id] > 0)
        coordinates = zip(indexes[0], indexes[1])  # convert to (y, x) coordinates point on matrix
        # search point around
        for coordinate in coordinates:
            y1 = max(0, coordinate[0] - search_radius)
            x1 = max(0, coordinate[1] - search_radius)
            y2 = min(self.memory_matrix_size[0], coordinate[0] + search_radius)
            x2 = min(self.memory_matrix_size[1], coordinate[1] + search_radius)
            # calculate weakening of cooling point
            weakening_cooling = 1 - np.count_nonzero(self.storage_matrix_memory[camera_id][y1:y2, x1:x2]) / max_points

            self.storage_matrix_memory[camera_id][coordinate] -= int(point_attenuation * weakening_cooling)

        return

    # шаг затухания
    def get_point_attenuation(self, camera_id):
        # getting quantity active object
        quantity_active = self.count_unique_object[camera_id]
        self.count_unique_object[camera_id] = 0

        # calculate relative quantity points
        relative_quantity = np.count_nonzero(self.storage_matrix_memory[camera_id]) / (self.memory_matrix_size[0] *
                                                                                       self.memory_matrix_size[1])
        #
        relative_quantity = min(1, (relative_quantity / 2) * (quantity_active /
                                                              (self.time_interval * 60 / self.frame_capture_interval)))

        # print("Cam_id: ", camera_id, " Quantity object: ", quantity_active,
        #       "  Attenuation:", int(self.max_point_attenuation * relative_quantity * self.time_interval))
        # attenuation step
        return int(self.max_point_attenuation * relative_quantity * self.time_interval)

    # проверка планироващика
    def check_schedule(self, camera_id):
        if self.time_schedule[camera_id] + timedelta(minutes=self.time_interval) < datetime.now():
            self.time_schedule[camera_id] = datetime.now()
            return True
        return False

    # сохранение активных объектов на матрице
    def save_point_object(self, camera_id, object_point):
        # x, y -> y, x
        # object_point = np.flip(object_point)
        # rescale point coordinates
        object_point = (object_point * self.memory_matrix_size).astype(int)
        # set points to matrix
        self.storage_matrix_memory[camera_id][object_point[:, 0], object_point[:, 1]] = 65535
        return

    # создать матрицу
    def get_pointed_matrix(self, point):
        matrix = np.zeros(self.memory_matrix_size, dtype=np.uint8)
        point = (self.get_median_point(point) * self.memory_matrix_size).astype(int)
        matrix[point[:, 0], point[:, 1]] = 255
        return matrix

    # медианная точка объекта
    @staticmethod
    def get_median_point(array_objects):

        point_array = np.dstack((np.absolute((array_objects[:, 1] - (array_objects[:, 1] - array_objects[:, 3]) / 2)),
                                 (np.absolute(array_objects[:, 0] - (array_objects[:, 0] - array_objects[:, 2]) / 2))))

        return np.squeeze(point_array, axis=0)

    # фильтрация трафика и парковки
    @staticmethod
    def search_active_objects(storage_objects, array_objects, static_object_time):

        new_static_object = []  # для добавления нового статичного объекта
        temporary_static = []
        time_processing = []

        # подготовка массива удаления вероятностей и класса объекта
        array_objects[:, -2] = 0
        array_objects[:, -1] = 0
        zeros_column = np.zeros((array_objects.shape[0], 1))
        array_objects = np.concatenate((array_objects, zeros_column), axis=1)
        # concatenate storage and new data соединение хранилища и новых данных
        work_array = np.vstack((storage_objects, array_objects))
        array_len = work_array.shape[0]

        # поиск уникальных объектов
        for n, item_ary in enumerate(work_array, 0):
            unique_object = True
            for slider in range(n + 1, array_len):
                if bbox_iou(item_ary[:-3], work_array[slider, :-3], True) > 0.80:  # 87 89 88 85 83 80 85
                    unique_object = False  # не уникальный
                    work_array[n, -3] += 1  # добавляю 1 если подобный объект существует
                    work_array[n, -1] += 1  # для расчета времени обработки объекта
                    if work_array[n, -3] < 1:
                        work_array[n, -3] = 1
                    work_array[n, -2] = 1  # ставлю 1 если объект не уникальный
                    work_array[slider, -2] = 1
                    # поиск нового статического объекта
                    if work_array[n, -3] > static_object_time:
                        if work_array[n, -3] < 50:
                            new_static_object.append(item_ary[:-3])
                            time_processing.append(work_array[n, -1])
                        work_array[n, -3] = 100  # гистерезис - защелка для фиксации
            if unique_object:
                if static_object_time >= work_array[n, -3] > 0:
                    temporary_static.append(np.copy(work_array[n]))
                work_array[n, -3] -= 2
                work_array[n, -1] += 1  # добавляю 1 для расчета времени обработки объекта
                # удаляю старые объекты
                if static_object_time < work_array[n, -3] < 50:
                    work_array[n, -3] = -1
        # получаю уникальные объекты
        new_unique_object = \
            np.vstack(
                (work_array[work_array[:, -2] == 0, :],
                 work_array[(0 < work_array[:, -3]) & (work_array[:, -3] < static_object_time), :]))
        # сохранение новых данных
        storage_objects = np.vstack((work_array[work_array[:, -2] == 0, :], work_array[work_array[:, -3] > 0, :]))
        storage_objects[:, -2] = 1

        return storage_objects, new_unique_object, np.array(new_static_object), time_processing, np.array(
            temporary_static)


def bbox_iou(boxA, boxB, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
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
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
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


if __name__ == "__main__":
    matrix_b = np.zeros((200, 200), dtype=np.uint8)
    matrix_g = np.zeros((200, 200), dtype=np.uint8)
    matrix_r = np.zeros((200, 200), dtype=np.uint8)

    matrix_b[30, 30] = 200
    matrix_g[20, 20] = 200
    matrix_r[50, 50] = 200

    matrix_t = np.zeros((200, 200), dtype=np.uint16)
    matrix_t[1, 1] = 65535
    print(np.array(matrix_t / 256, dtype=np.uint8))
    matrix_all = np.dstack((matrix_b, matrix_g, matrix_r))
    print(matrix_all.shape)
    cv2.imshow('image', matrix_all)
    cv2.waitKey(0)
