import numpy as np
from scipy.special import expit
import json
import os
from .config import NetworkConfig

class NeuralNetwork:
    def __init__(self, load_weights=True):
        self.inodes = NetworkConfig.INPUT_NODES
        self.hnodes = NetworkConfig.HIDDEN_NODES
        self.onodes = NetworkConfig.OUTPUT_NODES
        self.lr = NetworkConfig.LEARNING_RATE
        self.correct_predictions = 0
        self.total_predictions = 0
        
        if load_weights and self._weights_exist():
            self._load_weights()
        else:
            self._initialize_weights()
            
        self.activation_function = expit

    def _initialize_weights(self):
        """Инициализация весов"""
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    def _weights_exist(self):
        """Проверка существования файлов весов"""
        return (os.path.exists(NetworkConfig.WIH_WEIGHTS_FILE) and 
                os.path.exists(NetworkConfig.WHO_WEIGHTS_FILE))

    def _load_weights(self):
        """Загрузка весов из файлов"""
        try:
            with open(NetworkConfig.WIH_WEIGHTS_FILE, 'r') as f:
                self.wih = np.array(json.load(f))
            with open(NetworkConfig.WHO_WEIGHTS_FILE, 'r') as f:
                self.who = np.array(json.load(f))
        except Exception as e:
            print(f"Ошибка загрузки весов: {e}")
            self._initialize_weights()

    def _save_weights(self):
        """Сохранение весов в файлы"""
        os.makedirs(NetworkConfig.WEIGHTS_DIR, exist_ok=True)
        with open(NetworkConfig.WIH_WEIGHTS_FILE, 'w') as f:
            json.dump(self.wih.tolist(), f)
        with open(NetworkConfig.WHO_WEIGHTS_FILE, 'w') as f:
            json.dump(self.who.tolist(), f)

    def train(self, inputs_list, targets_list):
        """Обучение сети"""
        # Убедимся, что inputs_list - это одномерный массив
        if isinstance(inputs_list, (list, np.ndarray)) and len(inputs_list) > 0:
            if hasattr(inputs_list[0], '__len__'):  # Если это двумерный массив
                inputs_list = inputs_list[0]  # Берем первую строку
        
        # Убедимся, что targets_list - это одномерный массив
        if isinstance(targets_list, (list, np.ndarray)) and len(targets_list) > 0:
            if hasattr(targets_list[0], '__len__'):  # Если это двумерный массив
                targets_list = targets_list[0]  # Берем первую строку
        
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Прямой проход
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Вычисление ошибок
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # Обновление весов
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                    np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
                                    np.transpose(inputs))

        # Подсчет точности
        if np.argmax(final_outputs) == np.argmax(targets):
            self.correct_predictions += 1
        self.total_predictions += 1

    def query(self, inputs_list):
        """Запрос к сети"""
        # Убедимся, что inputs_list - это одномерный массив
        if isinstance(inputs_list, (list, np.ndarray)) and len(inputs_list) > 0:
            if hasattr(inputs_list[0], '__len__'):  # Если это двумерный массив
                inputs_list = inputs_list[0]  # Берем первую строку
                
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def get_accuracy(self):
        """Получение текущей точности"""
        if self.total_predictions == 0:
            return 0
        return self.correct_predictions / self.total_predictions

    def reset_accuracy(self):
        """Сброс счетчиков точности"""
        self.correct_predictions = 0
        self.total_predictions = 0