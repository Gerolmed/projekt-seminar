from numpy import ndarray
import pandas as pd


class Result:
    """The result after executing an algorithm"""

    def __init__(self, precision: ndarray, recall: ndarray, f1: float, confusion_matrix: pd.DataFrame,
                 algorithm_name: str, data_name: str, train_time: int, test_time: int):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.confusion_matrix = confusion_matrix
        self.algorithm_name = algorithm_name
        self.data_name = data_name
        self.train_time = train_time
        self.test_time = test_time
