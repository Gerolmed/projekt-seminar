from numpy import ndarray


class Result:
    """The result after executing an algorithm"""

    def __init__(self, precision: ndarray, recall: ndarray, f1: float):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
