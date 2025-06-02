# model_utils.py

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def remove_artifacts(image, mask):
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def pad_to_square(img):
    h, w = img.shape
    diff = abs(h - w)
    if h > w:
        pad_l = diff // 2
        pad_r = diff - pad_l
        return cv2.copyMakeBorder(img, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT)
    else:
        pad_t = diff // 2
        pad_b = diff - pad_t
        return cv2.copyMakeBorder(img, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT)

# def predict_mask(model, image_path, device='cuda', threshold=0.5):
#     model.eval()

#     image = cv2.imread(image_path, 0)  # grayscale
#     if image is None:
#         raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

#     # image_cropped, _ = remove_artifacts(image, np.zeros_like(image))
#     # image_padded = pad_to_square(image_cropped)
#     image_resized = cv2.resize(image, (512, 512))

#     cl_img = image_resized.astype(np.float32) / 255.0

#     transform = A.Compose([
#         A.Normalize(mean=[0.1454], std=[0.2365], max_pixel_value=1.0, p=1),
#         ToTensorV2()
#     ])
#     augmented = transform(image=cl_img)
#     input_tensor = augmented['image'].unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(input_tensor)
#     pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0] > threshold
#     pred_mask = (pred_mask * 255).astype(np.uint8)

#     return image_resized, pred_mask

# def predict_mask(model, image_np, device='cpu', threshold=0.5):
#     """
#     :param model: обученная модель
#     :param image_np: numpy array (grayscale), исходное изображение
#     :param device: 'cpu' или 'cuda'
#     :param threshold: порог бинаризации маски
#     :return: resized_image, pred_mask
#     """

#     # image_np — это уже numpy array, НЕ нужно читать через imread
#     image = image_np.copy()

#     # Обработка
#     image_cropped, _ = remove_artifacts(image, np.zeros_like(image))
#     image_padded = pad_to_square(image_cropped)
#     image_resized = cv2.resize(image_padded, (512, 512))

#     # image_resized = cv2.resize(image, (512, 512))

#     cl_img = image_resized.astype(np.float32) / 255.0

#     transform = A.Compose([
#         A.Normalize(mean=[0.1454], std=[0.2365], max_pixel_value=1.0, p=1),
#         ToTensorV2()
#     ])
#     augmented = transform(image=cl_img)
#     input_tensor = augmented['image'].unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(input_tensor)
#     pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0] > threshold
#     pred_mask = (pred_mask * 255).astype(np.uint8)

#     return image_resized, pred_mask

def predict_mask(model, image, device='cpu', threshold=0.5):
    # image — numpy array (grayscale)
    image_cropped, _ = remove_artifacts(image, np.zeros_like(image))
    image_padded = pad_to_square(image_cropped)
    image_resized = cv2.resize(image_padded, (512, 512))

    cl_img = image_resized.astype(np.float32) / 255.0

    transform = A.Compose([
        A.Normalize(mean=[0.1454], std=[0.2365], max_pixel_value=1.0, p=1),
        ToTensorV2()
    ])
    augmented = transform(image=cl_img)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0] > threshold
    pred_mask = (pred_mask * 255).astype(np.uint8)

    return image_resized, pred_mask

def draw_contour_on_image(image_resized, pred_mask):
    # Преобразуем изображение для отрисовки
    image_color = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_color, contours, -1, (255, 0, 0), thickness=2)

    return image_color