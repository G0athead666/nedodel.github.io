import os
import numpy as np
# Используем tensorflow.keras для импорта load_model и метрик
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError # Импортируем класс метрики MSE
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.mean_ = np.array([5.84333333, 3.054, 3.75866667, 1.19866667])
scaler.scale_ = np.array([0.82806613, 0.43214658, 1.76529823, 0.76223767])

script_dir = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(script_dir, '..', 'model', 'iris_model.h5')

print(f"DEBUG: Attempting to load model from: {model_path}")


custom_objects = {
    'mse': MeanSquaredError()
}

try:
    model = load_model(model_path, custom_objects=custom_objects)
    print("DEBUG: Model 'iris_model.h5' loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load model from {model_path}. Please check the path and custom_objects configuration.")
    print(f"ERROR DETAILS: {e}")
    raise # Перевызовем исключение, чтобы увидеть полный стек для дальнейшей отладки, если проблема не решена.


target_names = ["setosa", "versicolor", "virginica"]

def predict_iris(features: list):
    if not isinstance(features, list) or len(features) != 4:
        raise ValueError("Input features must be a list of 4 numerical values.")

    # Преобразование входных данных в NumPy массив и изменение формы для модели
    X = np.array(features).reshape(1, -1)

    # Применение StandardScaler
    X = scaler.transform(X)

    # Получение предсказаний от модели
    pred = model.predict(X)

    # Возвращаем название класса с наибольшей вероятностью
    return target_names[np.argmax(pred)]

