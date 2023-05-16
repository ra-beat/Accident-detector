import torch
import math
import numpy as np
from datetime import datetime, timedelta


# ========================================================================

class RoadAccidentFinder:

    def __init__(self, car_cls, min_static_time=80, frame_interval=10,
                 max_point_attenuation=2000, memory_matrix_size=(240, 430), image_resolution=1280):
        self.cameras_id = []
        self.service_name = 'AccidentDetector'
        # интервал кадра в секундах
        self.frame_interval = frame_interval
        # класс автомобиля, по умолчанию 2
        self.car_cls = car_cls
        # время обнаружения статического объекта
        self.static_object_time = min_static_time / self.frame_interval
        # размер матрицы памяти точек объекта
        self.memory_matrix_size = memory_matrix_size
        # максимальный шаг затухания точки за одну минуту
        self.max_point_attenuation = max_point_attenuation
        # входное разрешение координат массива для масштабирования
        self.input_resolution = \
            np.array([image_resolution, image_resolution / 1.777, image_resolution, image_resolution / 1.777, 1, 1])
        # хранилище времени жизни статических объектов
        self.storage_time_static = {}  # {camera_id: [time_1, time_2, ...time 20], ...}
        self.storage_time_length = 20
        # данные подключения камеры
        self.cameras_connection_data = []
        # размер матрицы памяти точек объекта
        self.memory_matrix_size = memory_matrix_size
        # хранилище данных матрицы
        self.storage_matrix_memory = {}
        # хранилище статических объектов
        self.storage_static_objects = {}  # {camera_id: object_array, ...}
        # таймер расписания (минуты)
        self.time_interval = 10
        self.time_schedule = {}

    def set_camera_connection_list(self, cameras_connection_data):
        # распакованный словарь данных для списка со значениями [[stream_id, stream_address], ...]
        cameras_connection_data = [list(x.values()) for x in cameras_connection_data]
        if self.cameras_connection_data != cameras_connection_data:
            self.cameras_id = [_[0] for _ in cameras_connection_data]  # извлечение id камеры
            self.cameras_connection_data = cameras_connection_data
            # Обновить данные о потоке обработки и удалить старые данные
            self.storage_matrix_memory = \
                {key: self.storage_matrix_memory[key]
                if key in self.storage_matrix_memory else np.zeros(self.memory_matrix_size, dtype=np.uint16)
                 for key in self.cameras_id}
            self.storage_static_objects = \
                {key: self.storage_static_objects[key] if key in self.storage_static_objects else np.empty((0, 6))
                 for key in self.cameras_id}
            self.time_schedule = \
                {key: self.time_schedule[key]
                if key in self.time_schedule else datetime.now() - timedelta(minutes=self.time_interval)
                 for key in self.cameras_id}
            self.storage_time_static = \
                {key: self.storage_time_static[key]
                if key in self.storage_time_static else [int(self.static_object_time / 2)] * self.storage_time_length
                 for key in self.cameras_id}

        return

    def road_accident_process(self, camera_id, prediction_data):
        # подготовить массив объектов - сохранить только объект автомобиля
        try:
            object_car = prediction_data[prediction_data[:, -1] == self.car_cls]
        except IndexError:
            return None

        accident_object = None  # для добавленных объектов аварии

        # if empty frame (car not found)
        if not torch.any(object_car):
            return accident_object

        # преобразовать pytorch в массив numpy и абсолютизировать координаты
        object_car = object_car.detach().numpy()
        object_car = object_car / self.input_resolution  # Преобразование в абсолютные координаты

        # фильтрованный массив объектов автомобиль
        self.storage_static_objects[camera_id], active_object, static_object, temporary_static = \
            self.search_active_objects(self.storage_static_objects[camera_id], object_car, self.static_object_time)
        # сохранить новые активные точки boxes.xyxy  # box with xyxy format, (N, 4)  для get_median_point
        self.save_point_object(camera_id, self.get_median_point(active_object))

        # проверить расписание
        if self.check_schedule(camera_id):
            # плавное исчезновение точек на матрице
            self.process_fading_points(camera_id, self.get_point_attenuation(camera_id))
        # проверить новый статический объект
        if static_object.any():
            for object_point in static_object:
                if self.check_status_object(camera_id, object_point):
                    accident_object = static_object
                else:
                    # удалить статический объект из матрицы памяти
                    self.delete_points(camera_id, object_point)

        return accident_object

    def delete_points(self, camera_id, point):
        # рассчитать размер объекта на матрице
        object_width, object_height, object_point = self.calculate_size_object(point)
        # рассчет координат
        y1 = int(max(0, object_point[0] - object_height / 2))
        x1 = int(max(0, object_point[1] - object_width / 2))
        y2 = int(min(self.memory_matrix_size[0], object_point[0] + object_height / 2))
        x2 = int(min(self.memory_matrix_size[1], object_point[1] + object_width / 2))
        # обнуление 2d среза статических объектов
        self.storage_matrix_memory[camera_id][y1:y2, x1:x2] = 0
        return

    # =================================================================================================================
    # ==================================== дописать процессор поиска аварий ===========================================

    # сохранить активную точку объекта в матрицу
    def save_point_object(self, camera_id, object_point):

        object_point = (object_point * self.memory_matrix_size).astype(int)
        # установить точки в матрицу
        self.storage_matrix_memory[camera_id][object_point[:, 0], object_point[:, 1]] = 65535
        return

    def get_point_attenuation(self, camera_id):
        # рассчитать относительное количество баллов
        relative_quantity = np.count_nonzero(self.storage_matrix_memory[camera_id]) / (self.memory_matrix_size[0] *
                                                                                       self.memory_matrix_size[1])
        # коэффициент предварительного усиления
        relative_quantity = min(1, relative_quantity * 10)
        # шаг затухания
        return int(self.max_point_attenuation * relative_quantity * self.time_interval)

    def calculate_size_object(self, object_point):
        object_width = int(abs(object_point[0] - object_point[2]) * self.memory_matrix_size[1])
        object_height = int(abs(object_point[1] - object_point[3]) * self.memory_matrix_size[0])
        # усредненная точка объекта
        object_point = (abs(object_point[1] - (object_point[1] - object_point[3]) / 2) * self.memory_matrix_size[0],
                        abs(object_point[0] - (object_point[0] - object_point[2]) / 2) * self.memory_matrix_size[1])
        return object_width, object_height, object_point

    def check_status_object(self, camera_id, object_point):
        # проверка позиции объекта
        min_num_point = 13
        # максимальный процент разности луча
        max_differences = 0.5
        # длина сканирующего луча
        length_beam = 3
        # рассчитать размер объекта на матрице
        object_width, object_height, object_point = self.calculate_size_object(object_point)
        half_size = min(object_width, object_height) / 2  # рассчитать половину максимального размера объекта
        # рассчитать координаты сканирующего луча
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

        # проверить положение объекта
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
            # проверка результата
            if count_a < min_num_point or count_b < min_num_point:  # not enough point
                continue
            # расчёт соотношения луча
            beam_ratio = 1 - min(count_a, count_b) / max(count_a, count_b)

            if beam_ratio < max_differences:  # соотношение находится в допустимых пределах
                return True
        return False  # авария не найдена

    # процесс выцветания точек
    def process_fading_points(self, camera_id, point_attenuation):
        # радиус для поиска соседних точек
        search_radius = 2
        max_points = (1 + search_radius + search_radius) ** 2
        # удаление меленьких значений
        self.storage_matrix_memory[camera_id][self.storage_matrix_memory[camera_id] < point_attenuation] = 0
        # активная точка поиска и декремент
        indexes = np.where(self.storage_matrix_memory[camera_id] > 0)
        coordinates = zip(indexes[0], indexes[1])  # преобразовать в (y, x) точку координат на матрице
        # точка поиска вокруг
        for coordinate in coordinates:
            y1 = max(0, coordinate[0] - search_radius)
            x1 = max(0, coordinate[1] - search_radius)
            y2 = min(self.memory_matrix_size[0], coordinate[0] + search_radius)
            x2 = min(self.memory_matrix_size[1], coordinate[1] + search_radius)
            # рассчитать ослабление точки охлаждения
            weakening_cooling = 1 - np.count_nonzero(self.storage_matrix_memory[camera_id][y1:y2, x1:x2]) / max_points

            self.storage_matrix_memory[camera_id][coordinate] -= int(point_attenuation * weakening_cooling)

        return

    @staticmethod
    def search_active_objects(storage_objects, array_objects, static_object_time):

        new_static_object = []  # для добавления нового статичного объекта
        temporary_static = []

        # слияние хранилища и новых данных
        work_array = np.vstack((storage_objects, array_objects))
        array_len = work_array.shape[0]

        # обработка массивов и поиск уникальных объектов
        for n, item_ary in enumerate(work_array, 0):
            unique_object = True
            for slider in range(n + 1, array_len):
                if bbox_iou(item_ary[:-2], work_array[slider, :-2], True) > 0.80:  # 87 89 88 85 83 80
                    unique_object = False  # объект не уникальный
                    work_array[n, -2] += 1  # добавить 1, если подобный объект найден
                    if work_array[n, -2] < 1:
                        work_array[n, -2] = 1
                    work_array[n, -1] = 1  # установить 1, если объект не уникален
                    work_array[slider, -1] = 1  # установить 1, если объект не уникален
                    # поиск новых статических объектов
                    if work_array[n, -2] > static_object_time:
                        if work_array[n, -2] < 50:
                            new_static_object.append(item_ary[:-2])
                        work_array[n, -2] = 100  # гистерезис - защелка для фиксации
            if unique_object:
                if static_object_time >= work_array[n, -2] > 0:
                    temporary_static.append(np.copy(work_array[n]))
                work_array[n, -2] -= 2
                # удаление старых статичных объектов
                if static_object_time < work_array[n, -2] < 50:
                    work_array[n, -2] = -1
        # получить уникальный объект
        new_unique_object = \
            np.vstack(
                (work_array[work_array[:, -1] == 0, :],
                 work_array[(0 < work_array[:, -2]) & (work_array[:, -2] < static_object_time), :]))
        # сохранить новые данные в хранилище
        storage_objects = np.vstack((work_array[work_array[:, -1] == 0, :], work_array[work_array[:, -2] > 0, :]))
        storage_objects[:, -1] = 1

        return storage_objects, new_unique_object, np.array(new_static_object), np.array(temporary_static)

    def check_schedule(self, camera_id):
        if self.time_schedule[camera_id] + timedelta(minutes=self.time_interval) < datetime.now():
            self.time_schedule[camera_id] = datetime.now()
            return True
        return False

    @staticmethod
    def get_median_point(array_objects):
        """calculating the midpoints of objects"""
        point_array = np.dstack((np.absolute((array_objects[:, 1] - (array_objects[:, 1] - array_objects[:, 3]) / 2)),
                                 (np.absolute(array_objects[:, 0] - (array_objects[:, 0] - array_objects[:, 2]) / 2))))

        return np.squeeze(point_array, axis=0)


# ========================================================================

# calculating the intersection over the union
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
