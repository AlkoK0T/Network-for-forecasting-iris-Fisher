"""
Конфигурационный файл для нейронной сети
"""

class NetworkConfig:
    """Конфигурация нейронной сети"""
    
    # Архитектура сети
    INPUT_NODES = 4      # Количество входных нейронов (признаки ириса)
    HIDDEN_NODES = 10    # Количество нейронов в скрытом слое
    OUTPUT_NODES = 3     # Количество выходных нейронов (классы ирисов)
    
    # Параметры обучения
    LEARNING_RATE = 0.1  # Скорость обучения
    EPOCHS = 100           # Количество эпох обучения
    TEST_SIZE = 0.2      # Доля тестовых данных
    
    # Пути к файлам
    DATA_PATH = "data/iris.csv"
    WEIGHTS_DIR = "weights"
    WIH_WEIGHTS_FILE = f"{WEIGHTS_DIR}/wih.json"
    WHO_WEIGHTS_FILE = f"{WEIGHTS_DIR}/who.json"
    
    # Названия классов
    CLASS_NAMES = {
        "Iris-setosa": [1, 0, 0],
        "Iris-versicolor": [0, 1, 0],
        "Iris-virginica": [0, 0, 1]
    }
    
    # Названия признаков
    FEATURE_NAMES = [
        "SepalLengthCm",
        "SepalWidthCm", 
        "PetalLengthCm",
        "PetalWidthCm"
    ]
    
    @classmethod
    def get_class_name(cls, one_hot_vector):
        """Получение названия класса по one-hot вектору"""
        for name, vector in cls.CLASS_NAMES.items():
            if vector == one_hot_vector:
                return name
        return "Unknown"

class GUIConfig:
    """Конфигурация графического интерфейса"""
    
    WINDOW_TITLE = "Training Progress"
    WINDOW_SIZE = "400x200"
    
    PROGRESS_BAR_LENGTH = 300
    
    # Цветовая схема
    BG_COLOR = "#f0f0f0"
    ACCENT_COLOR = "#4a90e2"
    
class TrainingConfig:
    """Конфигурация процесса обучения"""
    
    RANDOM_SEED = 42
    SHUFFLE_DATA = True
    SAVE_WEIGHTS = True
    SHOW_GUI = True