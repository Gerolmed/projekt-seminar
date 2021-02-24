from sklearn.neighbors import KNeighborsClassifier
from classifier.BaseClassifier import BaseClassifier


class KNearestNeighbor(BaseClassifier):

    def __init__(self):
        super().__init__("K-Nearest Neighbors", ["dict_vec_feature_set_data"])

    def construct_classifier(self):
        return KNeighborsClassifier(n_neighbors=5)
