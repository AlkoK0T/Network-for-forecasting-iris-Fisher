"""
Iris Flower Classification Neural Network Package
"""

from .neural_network import NeuralNetwork
from .data_processor import DataProcessor
from .trainer import TrainingGUI
from .config import NetworkConfig

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = ["NeuralNetwork", "DataProcessor", "TrainingGUI", "NetworkConfig"]