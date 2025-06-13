import streamlit as st
from datetime import datetime, date, timedelta
from PIL import Image
import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp

from model_set import predict_mask, draw_contour_on_image

st.set_page_config(page_title="Маммограмма", layout="wide")
# st.header("Анализ маммограммы") #title

if 'upload_image' not in st.session_state:
     st.session_state.upload_image = False

if 'show_predict' not in st.session_state:
     st.session_state.show_predict = False

if 'original_image_name' not in st.session_state:
    st.session_state.original_image_name = ''

if 'image_np' not in st.session_state:
    st.session_state.image_np = None

if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None

# if 'active_tab' not in st.session_state:
    # st.session_state.active_tab = "Загруженная маммограмма"
global new_size

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
    # model.eval()
    return model

model = load_model()


with st.sidebar:
    st.header("Загрузка изображения")
    uploaded_file = st.file_uploader(
        "Выберите изображение", 
        type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([2, 1]) 

with col1:
        # tab1, tab2 = st.tabs(["Загруженная маммограмма", "Предсказанная"])
        tab_options = ["Загруженная маммограмма", "Прогнозирование"]
        # default_index = tab_options.index(st.session_state.active_tab) if st.session_state.active_tab in tab_options else 0

        tab1, tab2 = st.tabs(tab_options)
        
        with tab1:
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('L')
                    original_width, original_height = image.size

                    new_size = (int(original_width / 0.83), int(original_height / 0.83))
                    resized_image = image.resize(new_size)

                    image_np = np.array(image)
                    st.session_state.image_np = image_np
                    
                    if st.session_state.original_image_name != uploaded_file.name:
                        st.session_state.original_image_name = uploaded_file.name
                        st.session_state.annotated_image = None
                        st.session_state.show_predict = False

                    st.image(resized_image,
                            # use_container_width=True
                            )
                    st.session_state.upload_image = True
                    
                    # if st.sidebar.button("Сделать прогноз"):
                    #     st.session_state.show_predict = True
                    #     # st.session_state.active_tab = "Предсказанная"
                    #     st.rerun()
                    with st.sidebar:
                        col1_, col2_, col3_ = st.columns([1, 3, 1])
                        with col2_:
                            predict = st.button("Сделать прогноз", 
                                                # use_container_width=True
                                                )
                        if predict:
                            st.session_state.show_predict = True
                                # st.session_state.active_tab = "Предсказанная"
                            st.rerun()   
                else:
                    st.info("Загрузите изображение")

        with tab2:
            if uploaded_file is None:
                if 'original_image_name' in st.session_state and st.session_state.original_image_name != '':
                    st.session_state.annotated_image = None
                    st.session_state.image_np = None
                    st.session_state.upload_image = False
                    st.session_state.show_predict = False
                    st.session_state.original_image_name = ''
                    # st.warning("Нет изображения для анализа")
                    # st.info("Изображение удалено")
                # else:
                st.warning("Нет изображения для анализа")

            elif st.session_state.show_predict:
                if st.session_state.image_np is not None:
                    if st.session_state.annotated_image is None:
                        with st.spinner("Выполняется анализ..."):
                            try:
                                image_resized, pred_mask = predict_mask(model, st.session_state.image_np, device='cpu')
                                annotated_image = draw_contour_on_image(image_resized, pred_mask)

                                # st.session_state.annotated_image = annotated_image

                                annotated_pil = Image.fromarray(annotated_image)
                                annotated_resized = annotated_pil.resize(new_size)  # <-- Применяем resize
                                st.session_state.annotated_image = np.array(annotated_resized)
                            except Exception as e:
                                 st.error(f"Ошибка при выполнении проноза: {e}")
                    if st.session_state.annotated_image is not None:
                        st.image(st.session_state.annotated_image, 
                            #  use_container_width=True, 
                            #  channels="BGR"
                             )
                    else:
                        st.warning("Не удалось загрузить прогноз")
            else:
                st.info("Выполните прогноз для анализа")


from datetime import datetime
import os

# Папка для сохранения
# SAVE_DIR = "saved_data"
# os.makedirs(SAVE_DIR, exist_ok=True)

# with col2:
#     st.markdown("### Информация о пациенте")

#     with st.form(key="patient_form"):
#         full_name = st.text_input("ФИО пациента")
#         birth_date = st.date_input("Дата рождения", value=None, format="DD/MM/YYYY")
#         visit_date = st.date_input("Дата приёма", value=None, format="DD/MM/YYYY")
#         description = st.text_area("Описание/заметки врача", height=200)

#         submit_button = st.form_submit_button(label="Сохранить данные")

#     if submit_button:
#         data = []
#         if full_name:
#             data.append(f"ФИО пациента: {full_name}")
#         if birth_date:
#             data.append(f"Дата рождения: {birth_date.strftime('%d-%m-%Y')}")
#         if visit_date:
#             data.append(f"Дата приёма: {visit_date.strftime('%d-%m-%Y')}")
#         if description:
#             data.append(f"Описание: {description}")

#         if data:
#             # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             # file_name = f"patient_{timestamp}.txt"
#             # file_path = os.path.join(SAVE_DIR, file_name)

#             # # Сохраняем файл ЛОКАЛЬНО на сервере
#             # with open(file_path, "w", encoding="utf-8") as f:
#             #     f.write("\n".join(data))

#             # st.success(f"Данные успешно сохранены: `{file_name}`")

#             # st.session_state.file_content = "\n".join(data)
#             # st.session_state.file_name = file_name
#             file_content = "\n".join(data)
#             file_name = f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

#             st.download_button(
#                 label="Скачать файл",
#                 data=file_content,
#                 file_name=file_name,
#                 mime="text/plain"
#             )
#         else:
#             st.warning("Нет данных для сохранения.")

#     # Эта часть ВНЕ формы — можно использовать download_button
#     # if 'file_content' in st.session_state and 'file_name' in st.session_state:
#     #     st.download_button(
#     #         label="Скачать файл",
#     #         data=st.session_state.file_content,
#     #         file_name=st.session_state.file_name,
#     #         mime="text/plain"
#     #     )

with col2:
    st.markdown("### Информация о пациенте")

    max_birth_date = date.today() - timedelta(days=365 * 100)
    min_birth_date = date.today() - timedelta(days=365 * 120)
    current_date = datetime.today().date()

    with st.form(key="patient_form"):
        full_name = st.text_input("ФИО пациента")
        birth_date = st.date_input(
             "Дата рождения", 
             value=None, 
             min_value=min_birth_date,
             max_value=max_birth_date,
             format="DD/MM/YYYY")
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

            # Сохраняем в session_state
            st.session_state.file_content = file_content
            st.session_state.file_name = file_name

            st.success(f"Данные сохранены. Вы можете скачать их ниже.")
        else:
            st.warning("Нет данных для сохранения.")

    # Всегда проверяем, есть ли что скачивать
    if 'file_content' in st.session_state and 'file_name' in st.session_state:
        st.download_button(
            label="Скачать файл",
            data=st.session_state.file_content,
            file_name=st.session_state.file_name,
            mime="text/plain"
        )
