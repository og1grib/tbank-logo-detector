import cv2
import numpy as np
from ultralytics import YOLO

from .schemas import DetectionResponse, Detection, BoundingBox, ErrorResponse
from .utils import draw_and_save_boxes

# Путь к весам модели
MODEL_PATH = "weights/best.pt"

# загрузка модели
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    model = None
    
    
def inference(file: bytes) -> DetectionResponse | ErrorResponse:
    
    # проверка корректности загрузки модели
    if model is None:
        return ErrorResponse(
            error="ModelNotLoaded",
            detail=f"Не удалось загрузить модель из {MODEL_PATH}"
        )
    
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # проверка корректности декодирования изображения
    if img is None:
        return ErrorResponse(
            error="InvalidFileFormat",
            detail="Файл не удалось декодировать. Возможно, повреждён."
        )
        
    # инференс модели
    try:
        res = model.predict(img, iou=0.5)
    except Exception as e:
        return ErrorResponse(
            error="ModelInferenceError",
            detail=f"Ошибка инференса модели: {str(e)}"
        )
        
    detections = []
    r = res[0]
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes.xyxy.cpu().numpy(): # [x1, y1, x2, y2] в yolo уже в нужном формате
            x1, y1, x2, y2 = map(int, box[:4])
            
            detections.append(Detection(bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)))
    
    # debug сохранение изображения с боксами в случае детекции
    # if detections:
        # draw_and_save_boxes(img, detections, save_path="debug/result.jpg")

    return DetectionResponse(detections=detections)

            
