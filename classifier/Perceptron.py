from classifier.BaseClassifier import BaseClassifier
from sklearn.linear_model import Perceptron


class LinearPerceptron(BaseClassifier):

    def __init__(self):
        super().__init__("Linear Perceptron", ["count_vec_data",
                                               "dict_vec_feature_set_data"
                                               ])

    def construct_classifier(self):
        return Perceptron(tol=1e-3, random_state=0)
