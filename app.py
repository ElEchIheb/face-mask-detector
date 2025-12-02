import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image  # لازم تكون فيها الدالة detect_and_predict_mask

# Page Config
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def mask_image():
    """معالجة الصورة المرفوعة"""
    global RGB_img
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model("mask_detector.model")
    
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_detection():
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg'])
        if image_file is not None:
            Image.open(image_file).save('./images/out.jpg')
            st.image(image_file, use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                mask_image()
                st.image(RGB_img, use_column_width=True)
    
    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])

        # load face detector & mask model
        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)
        model = load_model("mask_detector.model")
        
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not access the webcam.")
                break
            frame = cv2.resize(frame, (600, 500))
            (locs, preds) = detect_mask_image.detect_and_predict_mask(frame, net, model)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

mask_detection()
