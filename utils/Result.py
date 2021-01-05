from numpy import ndarray
import pandas as pd


class Result:
    """The result after executing an algorithm"""

    def __init__(self, precision: ndarray, recall: ndarray, f1: float, confusion_matrix: pd.DataFrame):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.confusion_matrix = confusion_matrix
