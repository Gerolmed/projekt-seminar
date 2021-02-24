from classifier.BaseClassifier import BaseClassifier
from sklearn.linear_model import LogisticRegression


class LogisticReg(BaseClassifier):

    def __init__(self):
        super().__init__("Logistic Regression", ["count_vec_data",
                                                 "dict_vec_feature_set_data"
                                                 ])

    def construct_classifier(self):
        return LogisticRegression(solver='liblinear', random_state=0)
