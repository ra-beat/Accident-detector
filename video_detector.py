import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
import cv2
from datetime import datetime, timedelta


# Resize image
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # изеняю размер и дополняю изображение
    shape = img.shape[:2]  # ткущий формат [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # соотношение масштабов (новое/ старое)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # уменьшение масштаба!!!
        r = min(r, 1.0)

    # вычисляю отступы
    ratio = r, r  # соотношение ширены и высоты
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # минимальный
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # растянутый
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # деление отсутпов на стороны
    dh /= 2

    if shape[::-1] != new_unpad:  # резайз
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # добавление рамки
    return img, ratio, (dw, dh)


def normalize_image(img0, device, half, img_size, stride):
    # плавное изменение размеров
    img = letterbox(img0, img_size, stride=stride)[0]

    # конвертирование
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR в RGB, в 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 в fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


class VideoDetector:

    def __init__(self, cuda_type, weights, image_size_input,
                 object_confidence_threshold, IOU_threshold_NMS,
                 filter_classes=None, image_resolution=1280, capture_interval=10):
        self.image_size_input = image_size_input
        self.object_confidence_threshold = object_confidence_threshold
        self.IOU_threshold_NMS = IOU_threshold_NMS
        self.filter_classes = filter_classes

        self.image_resolution = (int(image_resolution / 1.777), image_resolution, 3)
        self.scale = torch.tensor(self.image_resolution)[[1, 0, 1, 0]]
        self.img_size = (self.image_resolution[1], self.image_resolution[0])
        # захват кадров
        self.capture_interval = timedelta(seconds=capture_interval)
        self.stream_address = []  # [(camera_id, address), ...]
        self.scheduler_request_memory = []  # сохранить последний запрос stream_id [stream_id, ...]
        self.scheduler_last_run = datetime.now()  #время последнего запуска
        self.request_images = []

        # инициализация
        self.device = select_device(cuda_type)

        self.half = self.device.type != 'cpu'  # а у меня тольк ЦП

        # загрузка модели
        self.model = attempt_load(weights, map_location=self.device)  # загрузить модель FP32
        self.stride = int(self.model.stride.max())  # шаг модели
        image_size_input = check_img_size(image_size_input, s=self.model.stride.max())  # провека размера кадра
        if self.half:
            self.model.half()

        # получить имена модели
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # предварительный вывод
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, image_size_input, image_size_input).to(self.device).type_as(
                next(self.model.parameters())))

    # проверка графика получения кадра.
    # проверяети начинает захват кадра
    def check_capture_schedule(self):
        # получить текущий интервал
        if self.scheduler_last_run + self.capture_interval < datetime.now():
            self.scheduler_last_run = datetime.now()

            self.request_images = self.stream_address.copy()  # захват
        return


    # установка параметров камер
    def set_cameras(self, connection_data):
        self.stream_address = connection_data
        return

    # прогноз
    def __iter__(self):
        return self

    def __next__(self):
        # перезапуск планировщика списка кадров
        self.check_capture_schedule()

        camera_id = None
        img = None

        if self.request_images:
            camera_id, address = self.request_images.pop()  # получаем данные, рубим концы

            # полуаю кадр с камеры
            time_stop = 0
            capture = None
            try:
                # подключаюсь к камере
                capture = cv2.VideoCapture(address)
                while True:
                    status, img = capture.read()
                    time_stop += 1
                    if status or time_stop > 10:
                        break
            except Exception as er:
                print("Connection error:" + str(er))

            capture.release()
            img = cv2.resize(img, self.img_size)
            # проверябю изображение
            image = normalize_image(img, device=self.device, half=self.half, img_size=self.image_size_input,
                                    stride=self.stride)

            pred = self.model(image)[0]  # предсказываю объекты
            pred = non_max_suppression(pred, self.object_confidence_threshold, self.IOU_threshold_NMS)[0]
            return pred, camera_id, img, image.shape[2:]
        else:
            return None, None, None, None
