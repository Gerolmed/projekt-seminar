from algorithms.BaseAlgorithm import BaseAlgorithm
from sklearn.linear_model import Perceptron


class MultiLayerPerceptron(BaseAlgorithm):

    def __init__(self):
        super().__init__("Linear Perceptron", [# "count_vec_data",
                                                "dict_vec_pos_data"
                                                ])

    def construct_classifier(self):
        return Perceptron(tol=1e-3, random_state=0)
