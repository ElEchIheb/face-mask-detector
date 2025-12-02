# detect_mask_image.py
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detect_and_predict_mask(frame, faceNet, maskNet, confidence=0.5):
    """
    Input:
        frame: صورة (BGR)
        faceNet: شبكة الكشف عن الوجه (cv2.dnn)
        maskNet: موديل كشف الكمامة
        confidence: الحد الأدنى للاكتشاف
    Output:
        locs: قائمة bounding boxes [(startX, startY, endX, endY), ...]
        preds: قائمة التوقعات [(mask_prob, withoutMask_prob), ...]
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)
