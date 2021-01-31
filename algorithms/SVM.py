from sklearn.svm import LinearSVC
from algorithms.BaseAlgorithm import BaseAlgorithm


class SupportVectorMachine(BaseAlgorithm):

    def __init__(self):
        super().__init__("Linear Support Vector Machine", ["count_vec_data"])

    def construct_classifier(self):
        return LinearSVC(random_state=0)
