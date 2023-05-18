#pip install torchvision==0.10.1
#pip install torch==1.9.1
from utils.general import scale_coords
from images_storage import SaveImage
import torch
from video_detector import VideoDetector
from road_accident import RoadAccidentFinder
from dotenv import load_dotenv
import os
from telegram_notification import TelegramNotification


def processing():

    # обработка прогнозов, получение данных для обработки
    for det, camera_id, image, img_resize in detector:
        if camera_id is None:
            continue
        try:
            det[:, :4] = scale_coords(img_resize, det[:, :4], image.shape).round()
        except IndexError:
            det = torch.empty(0, 6)

        if not torch.any(det):  # ожидание новых данных
            continue

        # сохраняю тестовое изображение, только в режиме теста
        if save_test_image:
            save_image.show_detection_result(det, image, camera_id)

        # поиск аварий
        if (accident_object := road_accident_finder.road_accident_process(camera_id, det)) is not None:
            print("  Find accident!!!  for objects:  " + str(accident_object[0]))
            telegram_notification.send_image(image, accident_object[0], camera_id)
            save_image.save_road_accident_image(image, accident_object[0])
        if save_test_image:
            save_image.save_test_image(camera_id, road_accident_finder.get_test_map(camera_id))


if __name__ == '__main__':

    save_test_image = True  #False

    car_cls = 1

    cuda_type = "cpu"
    weights = "weights/yolov5l.pt"
    image_size_input = 640
    object_confidence_threshold = 0.5
    IOU_threshold_NMS = 0.45
    image_resolution = 1280
    capture_interval = 10

    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    bot_token = os.getenv('BOT_TOKEN')
    group_id = os.getenv('GROUP_ID')


    # отправляю кадр, для детектора
    detector = VideoDetector(
        cuda_type, weights, image_size_input,
        object_confidence_threshold, IOU_threshold_NMS,
        image_resolution, capture_interval)

    # сохраняю изображение с указанным размером
    save_image = SaveImage("/home/user/simple_road_accident/", "road_accident", "test_image", detector.names, car_cls, detector.scale, 1280)

    road_accident_finder = RoadAccidentFinder(car_cls=car_cls, min_static_time=80, frame_capture_interval=10, max_point_attenuation=2000)

    telegram_notification = TelegramNotification(bot_token, group_id)

    connection_data = [
            (90627, "https://s2.moidom-stream.ru/s/public/0000090627.m3u8"),
            # (1241, "https://s3.moidom-stream.ru/s/public/0000001241.m3u8"), # Куковское кольцо
            (192, "https://s3.moidom-stream.ru/s/public/0000000192.m3u8"),
            (1391, "https://s2.moidom-stream.ru/s/public/0000001391.m3u8"),
            (1993, "https://s2.moidom-stream.ru/s/public/0000001993.m3u8")
        ]

    detector.set_cameras(connection_data)
    road_accident_finder.set_camera_connection_list(connection_data)

    # запуск главного процесса
    while True:
        processing()




