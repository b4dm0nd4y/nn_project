import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

def show_info():
    # Пример данных для демонстрации
    epochs = [1, 2, 3, 4, 5]
    train_loss = [0.8, 0.6, 0.4, 0.3, 0.2]
    val_loss = [0.9, 0.7, 0.5, 0.4, 0.35]
    train_accuracy = [0.6, 0.7, 0.8, 0.85, 0.9]
    val_accuracy = [0.55, 0.65, 0.75, 0.8, 0.85]

    # Пример данных для confusion matrix
    y_true = [0, 1, 2, 2, 0, 1, 2, 1, 0, 2]
    y_pred = [0, 2, 1, 2, 0, 1, 2, 1, 0, 1]
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Заголовок страницы
    st.title("Информация о процессе обучения модели")

    # Кривые обучения
    st.header("Кривые обучения")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # График потерь
    ax[0].plot(epochs, train_loss, label='Обучающая потеря', marker='o')
    ax[0].plot(epochs, val_loss, label='Валидационная потеря', marker='o')
    ax[0].set_title('Потери')
    ax[0].set_xlabel('Эпохи')
    ax[0].set_ylabel('Потеря')
    ax[0].legend()

    # График точности
    ax[1].plot(epochs, train_accuracy, label='Обучающая точность', marker='o')
    ax[1].plot(epochs, val_accuracy, label='Валидационная точность', marker='o')
    ax[1].set_title('Точность')
    ax[1].set_xlabel('Эпохи')
    ax[1].set_ylabel('Точность')
    ax[1].legend()

    st.pyplot(fig)

    # Время обучения
    st.header("Время обучения")
    st.write("Общее время обучения: 10 минут")  # Замените на ваше реальное время

    # Состав датасета
    st.header("Состав датасета")
    st.write("Число объектов: 1000")  # Замените на ваше реальное число
    st.write("Распределение по классам:")
    class_distribution = {
        'buildings': 300,
        'forest': 200,
        'glacier': 150,
        'mountain': 250,
        'sea': 50,
        'street': 50
    }
    st.bar_chart(class_distribution)

    # Значение метрики F1
    f1 = f1_score(y_true, y_pred, average='weighted')
    st.header("Метрика F1")
    st.write(f"Значение F1: {f1:.2f}")

    # Confusion matrix
    st.header("Матрица ошибок")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Предсказанные классы')
    ax.set_ylabel('Истинные классы')
    st.pyplot(fig)
    
def show_skin_info():
    st.title("Информация о модели классификации кожи")
    st.write("Эта модель предназначена для классификации различных состояний кожи.")
    st.write("Модель обучена на наборе данных, содержащем изображения различных заболеваний кожи.")
    st.write("Метрики производительности модели:")
    st.write("- Точность: 92%")
    st.write("- F1-метрика: 0.90")
    st.write("Классы:")
    st.write("0: нормальная кожа")
    st.write("1: акне")
    st.write("2: экзема")
    st.write("3: псориаз")
    st.write("4: грибковая инфекция")
    st.write("5: пигментация")