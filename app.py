import streamlit as st
from PIL import Image
import requests
import torch
from torchvision import transforms, models
from torch import nn
from io import BytesIO
import time
from info import show_info, show_skin_info  # Импортируем функции из info.py

# Конфигурация
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
labels = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

# Определение преобразований для входного изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Загрузка модели
@st.cache_resource
def load_model():
    model = models.resnet50(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 6)
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    return model

model = load_model()
checkpoint_path = './models/model1.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

st.title("Многостраничное приложение для классификации изображений")

# Меню для навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:", ["Классификация изображений", "Информация о модели", "Классификация кожи", "Информация о модели кожи"])

def classify_image(url):
    try:
        start_time = time.time()
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_batch)
        _, predicted = torch.max(output, 1)
        predicted_label = labels[predicted.item()]
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Предсказанный класс: {predicted_label}, Время ответа: {elapsed_time:.2f} секунд")
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения по URL '{url}': {e}")

if page == "Классификация изображений":
    st.header("Классификация изображений по URL")
    urls = [st.text_input(f"Введите URL изображения {i + 1}:") for i in range(3)]
    for url in urls:
        if url:
            classify_image(url)

elif page == "Информация о модели":
    show_info()

elif page == "Классификация кожи":
    st.header("Классификация изображений кожи по URL")
    skin_urls = [st.text_input(f"Введите URL изображения кожи {i + 1}:") for i in range(3)]
    for url in skin_urls:
        if url:
            classify_image(url)  # Здесь нужно будет изменить на функцию для классификации кожи

elif page == "Информация о модели кожи":
    show_skin_info()