import cv2
import torch

from deep_sort_realtime.deepsort_tracker import DeepSort

window_name = 'img'
source = "video_name.mp4"

#класс yolov5
class YoloDetector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #загрузка модели
    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        return model

    #получение результатов детекции
    def get_results(self, frame):
        self.model.to(self.device)
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    #преобразование результатов
    def plot_boxes(self, results, height, width):
        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)

            confidence = float(row[4].item())
            detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], confidence, labels[i]))

        return detections


cap = cv2.VideoCapture(source)

#фул скрин
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#создание объектов детектора yolo и трекера deepsort
detector = YoloDetector(model_name="best_6.0.pt")

object_tracker = DeepSort(max_age=5,
                          n_init=2,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

while cap.isOpened():
    succes, img = cap.read()

    try:
        if succes == False:
            raise Exception("Не удалось открыть файл")
    except Exception as e:
        print(e)

    results = detector.get_results(img)
    detections = detector.plot_boxes(results, height=img.shape[0], width=img.shape[1])
    track = object_tracker.update_tracks(detections, frame=img)

    for track in track:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

    cv2.imshow(window_name, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
