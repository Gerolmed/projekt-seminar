from sklearn.naive_bayes import MultinomialNB
from algorithms.BaseAlgorithm import BaseAlgorithm


class NaiveBayes(BaseAlgorithm):

    def __init__(self):
        super().__init__("naive_bayes", ["dict_vec_pos_data"])

    def construct_classifier(self):
        return MultinomialNB(alpha=0.01)
