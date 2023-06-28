import io
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms, models
from torch import nn
import cv2

# MODEL_PATH = './CACD_r50_improved_1.pth'
MODEL_PATH = './FGNet73_GAN_SSL_ckpt.pth'
LABELS_PATH = './FGNet_labels.txt'


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model(model_path):
    model = models.resnet18()
    num_ftrs = model.fc.in_features  
    model.fc = nn.Sequential(
               nn.Linear(num_ftrs, 512),
               nn.ReLU(inplace=True),
               nn.Dropout(0.4),
               nn.Linear(512, 82))
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
        for i in range(all_prob.size(0)):
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
        st.write('Calculating results...')
        predict(model, categories, image)

if __name__ == '__main__':
    main()