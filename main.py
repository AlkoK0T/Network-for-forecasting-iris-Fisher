import numpy as np
import random
from src.neural_network import NeuralNetwork
from src.data_processor import DataProcessor
from src.trainer import TrainingGUI
from src.config import NetworkConfig, TrainingConfig

def main():
    # Инициализация
    processor = DataProcessor()
    network = NeuralNetwork(load_weights=False)
    
    # Загрузка и подготовка данных
    df = processor.load_data(NetworkConfig.DATA_PATH)
    print("Структура данных:")
    print(df.head())
    print("Колонки:", df.columns.tolist())
    print("Уникальные значения Species:", df['Species'].unique())
    
    X, y = processor.prepare_data(df)
    print("Размерность X:", X.shape)
    print("Размерность первого элемента y:", len(y[0]) if y else "пусто")
    print("Пример y[0]:", y[0] if y else "пусто")
    
    X_normalized = processor.normalize_data(X)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = processor.split_data(
        X_normalized, y, test_size=NetworkConfig.TEST_SIZE, 
        random_state=TrainingConfig.RANDOM_SEED)
    
    print("Размерность X_train:", X_train.shape)
    print("Размерность y_train[0]:", len(y_train[0]) if y_train else "пусто")
    print("Тип y_train[0]:", type(y_train[0]) if y_train else "пусто")
    
    # Обучение с GUI
    if TrainingConfig.SHOW_GUI:
        gui = TrainingGUI(NetworkConfig.EPOCHS, len(X_train))
        
        for epoch in range(NetworkConfig.EPOCHS):
            # Перемешивание данных
            if TrainingConfig.SHUFFLE_DATA:
                indices = list(range(len(X_train)))
                random.shuffle(indices)
            else:
                indices = range(len(X_train))
            
            for i, idx in enumerate(indices):
                # Добавим проверку размерностей
                x_sample = X_train[idx]
                y_sample = y_train[idx]
                
                if len(x_sample) != NetworkConfig.INPUT_NODES:
                    print(f"Ошибка: размерность входных данных {len(x_sample)}, ожидалось {NetworkConfig.INPUT_NODES}")
                    continue
                    
                if len(y_sample) != NetworkConfig.OUTPUT_NODES:
                    print(f"Ошибка: размерность выходных данных {len(y_sample)}, ожидалось {NetworkConfig.OUTPUT_NODES}")
                    print(f"Значение: {y_sample}")
                    continue
                
                network.train(x_sample, y_sample)
                accuracy = network.get_accuracy()
                gui.update_progress(epoch + 1, i + 1, accuracy)
            
            network.reset_accuracy()
        
        # Сохранение весов
        if TrainingConfig.SAVE_WEIGHTS:
            network._save_weights()
        
        gui.destroy()
    
    # Тестирование
    scorecard = []
    for i in range(len(X_test)):
        outputs = network.query(X_test[i])
        predicted_label = np.argmax(outputs)
        actual_label = np.argmax(y_test[i])
        
        if predicted_label == actual_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    
    scorecard_array = np.array(scorecard)
    print(f"Эффективность = {scorecard_array.sum() / scorecard_array.size}")

if __name__ == "__main__":
    main()