import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Görüntü İşleme & Filtreleme Paneli", layout="wide")
st.title("Görüntü Filtreleme ve Analiz Paneli")

uploaded_file = st.file_uploader(
    "Bir görsel yükleyin", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Sidebar Filtre Seçim Grubu
    st.sidebar.header("Filtre & Algoritma Ayarları")
    category = st.sidebar.selectbox(
        "Kategori",
        [
            "Renk Uzayları & Tonlama",
            "Bulanıklaştırma (Blurring / Smoothing)",
            "Keskinleştirme (Sharpening)",
            "Kenar Tespiti (Edge Detection)",
            "Eşikleme (Thresholding)",
            "Morfolojik Operasyonlar",
        ],
    )

    processed_img = img_array.copy()

    if category == "Renk Uzayları & Tonlama":
        mode = st.sidebar.selectbox(
            "Yöntem",
            [
                "Luminance Gray (Weighted)",
                "HSV - Hue",
                "HSV - Saturation",
                "HSV - Value",
                "Sepia",
            ],
        )
        if mode == "Luminance Gray (Weighted)":
            processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif mode.startswith("HSV"):
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            idx = {"HSV - Hue": 0, "HSV - Saturation": 1, "HSV - Value": 2}[mode]
            processed_img = hsv[:, :, idx]
        elif mode == "Sepia":
            kernel = np.array(
                [
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131],
                ]
            )
            sepia = cv2.transform(img_array, kernel)
            processed_img = np.clip(sepia, 0, 255).astype(np.uint8)

    elif category == "Bulanıklaştırma (Blurring / Smoothing)":
        mode = st.sidebar.selectbox(
            "Filtre Tipi",
            [
                "Gaussian Blur",
                "Box Filter (Average)",
                "Median Blur",
                "Bilateral Filter",
            ],
        )
        if mode == "Gaussian Blur":
            ksize = st.sidebar.slider("Kernel Size (Tek sayı)", 3, 31, 5, step=2)
            sigma = st.sidebar.slider("SigmaX", 0.1, 10.0, 1.0)
            processed_img = cv2.GaussianBlur(img_array, (ksize, ksize), sigma)
        elif mode == "Box Filter (Average)":
            ksize = st.sidebar.slider("Kernel Size (Tek sayı)", 3, 31, 5, step=2)
            processed_img = cv2.blur(img_array, (ksize, ksize))
        elif mode == "Median Blur":
            ksize = st.sidebar.slider("Kernel Size (Tek sayı)", 3, 31, 5, step=2)
            processed_img = cv2.medianBlur(img_array, ksize)
        elif mode == "Bilateral Filter":
            d = st.sidebar.slider("Diameter (d)", 3, 15, 9)
            sigma_color = st.sidebar.slider("Sigma Color", 10, 150, 75)
            sigma_space = st.sidebar.slider("Sigma Space", 10, 150, 75)
            processed_img = cv2.bilateralFilter(
                img_array, d, sigma_color, sigma_space
            )

    elif category == "Keskinleştirme (Sharpening)":
        mode = st.sidebar.selectbox(
            "Keskinleştirme Yöntemi",
            [
                "Standart Kernel Sharpening",
                "Aşırı Keskinleştirme (Excessive)",
                "Unsharp Masking",
            ],
        )
        if mode == "Standart Kernel Sharpening":
            strength = st.sidebar.slider("Güç / Ağırlık", 1, 5, 1)
            # Merkezdeki ağırlığı kullanıcı seçimine göre dinamik ayarlayan matris
            kernel = np.array(
                [
                    [0, -1, 0],
                    [-1, 4 + strength, -1],
                    [0, -1, 0],
                ]
            )
            sharpened = cv2.filter2D(img_array, -1, kernel)
            processed_img = np.clip(sharpened, 0, 255).astype(np.uint8)

        elif mode == "Aşırı Keskinleştirme (Excessive)":
            kernel = np.array(
                [
                    [-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1],
                ]
            )
            sharpened = cv2.filter2D(img_array, -1, kernel)
            processed_img = np.clip(sharpened, 0, 255).astype(np.uint8)

        elif mode == "Unsharp Masking":
            sigma = st.sidebar.slider("Blur Sigma", 0.5, 10.0, 1.5)
            amount = st.sidebar.slider("Keskinlik Kat sayısı (Amount)", 0.1, 3.0, 1.0)
            
            blurred = cv2.GaussianBlur(img_array, (0, 0), sigma)
            # Formula: original + amount * (original - blurred)
            sharpened = cv2.addWeighted(img_array, 1.0 + amount, blurred, -amount, 0)
            processed_img = np.clip(sharpened, 0, 255).astype(np.uint8)

    elif category == "Kenar Tespiti (Edge Detection)":
        mode = st.sidebar.selectbox(
            "Algoritma", ["Canny", "Sobel Combined", "Laplacian"]
        )
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        if mode == "Canny":
            t1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
            t2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
            processed_img = cv2.Canny(gray, t1, t2)
        elif mode == "Sobel Combined":
            ksize = st.sidebar.selectbox("Kernel Size", [3, 5, 7], index=0)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            processed_img = cv2.magnitude(sobelx, sobely)
            processed_img = np.uint8(np.clip(processed_img, 0, 255))
        elif mode == "Laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            processed_img = np.uint8(np.absolute(lap))

    elif category == "Eşikleme (Thresholding)":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mode = st.sidebar.selectbox(
            "Yöntem", ["Binary", "Otsu Automatic", "Adaptive Gaussian"]
        )

        if mode == "Binary":
            thresh = st.sidebar.slider("Eşik Değeri", 0, 255, 127)
            _, processed_img = cv2.threshold(
                gray, thresh, 255, cv2.THRESH_BINARY
            )
        elif mode == "Otsu Automatic":
            _, processed_img = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif mode == "Adaptive Gaussian":
            block_size = st.sidebar.slider("Block Size", 3, 31, 11, step=2)
            c = st.sidebar.slider("Constant C", 0, 20, 2)
            processed_img = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c,
            )

    elif category == "Morfolojik Operasyonlar":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        op = st.sidebar.selectbox(
            "Operasyon",
            ["Dilation (Genişletme)", "Erosion (Aşındırma)", "Opening", "Closing"],
        )
        ksize = st.sidebar.slider("Structuring Element Size", 3, 21, 5, step=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

        op_map = {
            "Dilation (Genişletme)": cv2.MORPH_DILATE,
            "Erosion (Aşındırma)": cv2.MORPH_ERODE,
            "Opening": cv2.MORPH_OPEN,
            "Closing": cv2.MORPH_CLOSE,
        }
        processed_img = cv2.morphologyEx(binary, op_map[op], kernel)

    # Görselleştirme Ekranı
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orijinal Görsel")
        st.image(img_array, use_container_width=True)

    with col2:
        st.subheader(f"İşlenmiş Görsel ({category})")
        st.image(processed_img, use_container_width=True)
