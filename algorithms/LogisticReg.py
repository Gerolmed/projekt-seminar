from algorithms.BaseAlgorithm import BaseAlgorithm
from sklearn.linear_model import LogisticRegression


class LogisticReg(BaseAlgorithm):

    def __init__(self):
        super().__init__("Logistic Regression", [# "count_vec_data",
                                                "dict_vec_pos_data"
                                                ])

    def construct_classifier(self):
        return LogisticRegression(solver='liblinear', random_state=0)
