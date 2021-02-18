from sklearn.svm import LinearSVC
from classifier.BaseClassifier import BaseClassifier


class SupportVectorMachine(BaseClassifier):

    def __init__(self):
        super().__init__("Linear Support Vector Machine", ["count_vec_data",
                                                           "dict_vec_feature_set_data"
                                                           ])

    def construct_classifier(self):
        return LinearSVC(random_state=0, max_iter=100000)
