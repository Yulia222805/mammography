import streamlit as st
from datetime import datetime
from PIL import Image
import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp

from model_set import predict_mask, draw_contour_on_image

st.set_page_config(page_title="Маммограмма", layout="wide")

# Загрузка модели
@st.cache_resource
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b2",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load('unet_best_7828.pth', map_location=device))
    model = model.to(device)
    return model

model = load_model()

# Переменные для временного хранения
upload_image = False
show_predict = False
original_image_name = ''
image_np = None
annotated_image = None
new_size = None

with st.sidebar:
    st.header("Загрузка изображения")
    uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file is not None:
        # Открываем изображение напрямую из загруженного файла
        image = Image.open(uploaded_file).convert('L')  # grayscale
        image_np = np.array(image)  # преобразуем в numpy array

        st.image(image, caption="Загруженная маммограмма", use_container_width=True)

        if st.sidebar.button("🔮 Инференс"):
            with st.spinner("Выполняется анализ..."):
                try:
                    # Теперь мы передаём изображение как numpy array, а не путь к файлу
                    image_resized, pred_mask = predict_mask(model, image=image_np, device='cpu')
                    annotated_image = draw_contour_on_image(image_resized, pred_mask)

                    st.markdown("###### Результат анализа:")
                    st.image(annotated_image, use_container_width=True, channels="BGR")

                except Exception as e:
                    st.error(f"Ошибка при выполнении инференса: {e}")

with col2:
    st.markdown("### Информация о пациенте")

    with st.form(key="patient_form"):
        full_name = st.text_input("ФИО пациента")
        birth_date = st.date_input("Дата рождения", value=None, format="DD/MM/YYYY")
        visit_date = st.date_input("Дата приёма", value=None, format="DD/MM/YYYY")
        description = st.text_area("Описание/заметки врача", height=200)

        submit_button = st.form_submit_button(label="Сохранить данные")

    if submit_button:
        data = []
        if full_name:
            data.append(f"ФИО пациента: {full_name}")
        if birth_date:
            data.append(f"Дата рождения: {birth_date.strftime('%d-%m-%Y')}")
        if visit_date:
            data.append(f"Дата приёма: {visit_date.strftime('%d-%m-%Y')}")
        if description:
            data.append(f"Описание: {description}")

        if data:
            file_content = "\n".join(data)
            file_name = f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.session_state.file_content = file_content
            st.session_state.file_name = file_name

            st.success(f"Данные сохранены. Вы можете скачать их ниже.")
        else:
            st.warning("Нет данных для сохранения.")

    # Проверяем наличие данных для скачивания
    if 'file_content' in locals() or 'file_content' in globals():
        st.download_button(
            label="Скачать файл",
            data=file_content,
            file_name=file_name,
            mime="text/plain"
        )