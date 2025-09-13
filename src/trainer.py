import tkinter as tk
from tkinter import ttk
from .config import GUIConfig

class TrainingGUI:
    def __init__(self, epochs, total_records):
        self.root = tk.Tk()
        self.root.title(GUIConfig.WINDOW_TITLE)
        self.root.geometry(GUIConfig.WINDOW_SIZE)
        self.root.configure(bg=GUIConfig.BG_COLOR)
        
        self.epoch_var = tk.IntVar()
        self.record_var = tk.IntVar()
        self.accuracy_var = tk.DoubleVar()
        
        self.setup_ui(epochs, total_records)
        
    def setup_ui(self, epochs, total_records):
        """Настройка интерфейса"""
        # Стилизация
        style = ttk.Style()
        style.theme_use('clam')
        
        # Прогресс эпох
        tk.Label(self.root, text="Epoch:", bg=GUIConfig.BG_COLOR).grid(
            row=0, column=0, sticky='w', padx=10, pady=5)
        self.epoch_progress = ttk.Progressbar(
            self.root, orient="horizontal", maximum=epochs, 
            variable=self.epoch_var, length=GUIConfig.PROGRESS_BAR_LENGTH)
        self.epoch_progress.grid(row=0, column=1, columnspan=3, padx=10, pady=5, sticky='ew')
        self.epoch_label = tk.Label(self.root, textvariable=self.epoch_var, bg=GUIConfig.BG_COLOR)
        self.epoch_label.grid(row=0, column=4, padx=10, pady=5)
        
        # Прогресс записей
        tk.Label(self.root, text="Records:", bg=GUIConfig.BG_COLOR).grid(
            row=1, column=0, sticky='w', padx=10, pady=5)
        self.record_progress = ttk.Progressbar(
            self.root, orient="horizontal", maximum=total_records, 
            variable=self.record_var, length=GUIConfig.PROGRESS_BAR_LENGTH)
        self.record_progress.grid(row=1, column=1, columnspan=3, padx=10, pady=5, sticky='ew')
        self.record_label = tk.Label(self.root, textvariable=self.record_var, bg=GUIConfig.BG_COLOR)
        self.record_label.grid(row=1, column=4, padx=10, pady=5)
        
        # Точность
        tk.Label(self.root, text="Accuracy:", bg=GUIConfig.BG_COLOR).grid(
            row=2, column=0, sticky='w', padx=10, pady=5)
        self.accuracy_label = tk.Label(self.root, textvariable=self.accuracy_var, 
                                     bg=GUIConfig.BG_COLOR, fg=GUIConfig.ACCENT_COLOR)
        self.accuracy_label.grid(row=2, column=1, padx=10, pady=5)
        
        # Конфигурация сетки
        self.root.columnconfigure(1, weight=1)
        
    def update_progress(self, epoch, record, accuracy):
        """Обновление прогресса"""
        self.epoch_var.set(epoch)
        self.record_var.set(record)
        self.accuracy_var.set(round(accuracy, 4))
        self.root.update()
        
    def show(self):
        """Отображение окна"""
        self.root.mainloop()
        
    def destroy(self):
        """Закрытие окна"""
        self.root.destroy()