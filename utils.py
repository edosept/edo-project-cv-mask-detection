import cv2
import numpy as np


def detect_face(net, frame, conf_threshold=0.7):
    # Siapkan input image
    h, w, c = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    # Feedforward
    #     prediksi dari SSD mengeluarkan output (1, 1, N, 7)
    #     7 output tersebut merupakan: image_id, label, conf, x_min, y_min, x_max, y_max    
    net.setInput(blob)
    detections = net.forward()

    # Filter prediksi yang low confidence
    bbox = []
    for _, _, conf, x1, y1, x2, y2 in detections[0, 0]:
        if conf > conf_threshold:
            box = np.array([x1, y1, x2, y2]) * [w, h, w, h]
            bbox.append(box.astype(int))
    return bbox

def normalize_image(img):
    mean = img.reshape(-1, 3).mean(0).reshape(1, 1, -1)
    std = img.reshape(-1, 3).std(0).reshape(1, 1, -1)

    img = (img - mean) / std
    img = (np.clip(img, [-4, -4, -4], [4, 4, 4]) + 4) / 8
    img = (img*255).astype(np.uint8)
    return img

def calculate_skin_percent(face, min_val=(110, 125, 125), max_val=(180, 140, 135)):
    face = normalize_image(face)
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    min_val = np.array(min_val, dtype=np.uint8)
    max_val = np.array(max_val, dtype=np.uint8)

    skin = ((ycrcb >= min_val) & (ycrcb <= max_val)).all(2)
    skin_percent = skin.mean()
    return skin_percent