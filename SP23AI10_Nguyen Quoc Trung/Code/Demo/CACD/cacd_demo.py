import matplotlib.pyplot as plt
import io
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms, models
from torch import nn
import cv2, dlib
import numpy as np
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

MODEL_PATH = './CACD_r50_improved_1.pth'
# MODEL_PATH = './CACD_R50_checkpoint.pth'
LABELS_PATH = './cacd_label.txt'

# Face detection and alignment
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# def load_image():
#     uploaded_file = st.file_uploader(label='Pick an image to test')
#     if uploaded_file is not None:
#         # Convert the file to an opencv image.
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         opencv_image = cv2.imdecode(file_bytes, 1)
#         st.image(opencv_image, channels="BGR")
#         return opencv_image
#     else:
#         return None
def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def detect_face(image):
    image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    det = detector(image, 1)[0]
    height, width = image.shape[:2]
    shape = predictor(image, det)
    left_eye = extract_left_eye_center(shape)
    right_eye = extract_right_eye_center(shape)

    M = get_rotation_matrix(left_eye, right_eye)
    rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC)

    cropped = crop_image(rotated, det)
    return cropped


def load_model(model_path):
    model = models.resnet50()
    num_ftrs = model.fc.in_features  
    model.fc = nn.Sequential(
               nn.Linear(num_ftrs, 512),
               nn.ReLU(inplace=True),
               nn.Dropout(0.4),
               nn.Linear(512, 2000))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        all_prob, all_catid = torch.topk(probabilities, len(categories))
        # for i in range(all_prob.size(0)):
        for i in range(10):
            st.write(categories[all_catid[i]], all_prob[i].item())
    except:
        st.write('No image found.')


def main():
    st.title('Model demo')
    model = load_model(MODEL_PATH)
    categories = load_labels(LABELS_PATH)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Detected face')
        face = detect_face(image)
        face = Image.fromarray(np.uint8(face)).convert('RGB')
        st.image(face)
        predict(model, categories, face)

if __name__ == '__main__':
    main()