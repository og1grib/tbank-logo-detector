import os
import cv2

def draw_and_save_boxes(img, detections, save_path="debug/last_result.jpg"):
    img_draw = img.copy()

    for det in detections:
        x1, y1 = det.bbox.x_min, det.bbox.y_min
        x2, y2 = det.bbox.x_max, det.bbox.y_max
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, "T-Bank Logo", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_draw)
