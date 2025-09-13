import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.label_map = {
            "Iris-setosa": [1, 0, 0],
            "Iris-versicolor": [0, 1, 0], 
            "Iris-virginica": [0, 0, 1]
        }
        self.reverse_label_map = {tuple(v): k for k, v in self.label_map.items()}

    def load_data(self, filepath):
        """Загрузка данных из CSV"""
        df = pd.read_csv(filepath)
        return df

    def prepare_data(self, df):
        """Подготовка данных для обучения"""
        # Определяем колонки с признаками (исключаем Id и Species)
        feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        
        # Извлекаем признаки
        X = df[feature_columns].values
        
        # Извлекаем метки и преобразуем в one-hot
        y_labels = df['Species'].str.strip()  # Убираем пробелы
        y = [self.label_map[label] for label in y_labels]
        
        return X, y

    def normalize_data(self, X, fit_scaler=True):
        """Нормализация данных"""
        if fit_scaler:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Разделение данных на тренировочные и тестовые"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def get_label_name(self, one_hot_vector):
        """Получение названия класса по one-hot вектору"""
        return self.reverse_label_map.get(tuple(one_hot_vector), "Unknown")