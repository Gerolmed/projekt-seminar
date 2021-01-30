from sklearn.neighbors import KNeighborsClassifier
from algorithms.BaseAlgorithm import BaseAlgorithm


class KNearestNeighbor(BaseAlgorithm):

    def __init__(self):
        super().__init__("Multinomial Naive Bayes", ["dict_vec_pos_data"])

    def construct_classifier(self):
        return KNeighborsClassifier(n_neighbors=3)
