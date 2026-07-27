import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Streamlit Görsel Grileştirme Aracı")

uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Yüklenen dosyayı PIL Image üzerinden NumPy dizisine çevir (RGB)
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # RGB -> GRAY dönüşümü
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Yan yana gösterim
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orijinal Görsel")
        st.image(img_array, use_container_width=True)
    with col2:
        st.subheader("Gri Tonlamalı Görsel")
        st.image(gray_image, use_container_width=True)