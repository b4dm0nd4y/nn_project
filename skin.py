import streamlit as st
from PIL import Image
import requests
import torch
from torchvision import transforms,  models
from io import BytesIO

# Конфигурация
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
labels = {
    0: 'нормальная кожа',
    1: 'акне',
    2: 'экзема',
    3: 'псориаз',
    4: 'грибковая инфекция',
    5: 'пигментация'
}

# Определение преобразований для входного изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Загрузка модели (предполагается, что модель уже обучена)
@st.cache_resource
def load_skin_model():
    model = models.resnet50(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(2048, 1)
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    return model

model = load_skin_model()
checkpoint_path = './models/model2.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

def classify_skin_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_batch)
    _, predicted = torch.max(output, 1)
    return labels[predicted.item()], image