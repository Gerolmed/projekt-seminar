from sklearn.naive_bayes import MultinomialNB
from algorithms.BaseAlgorithm import BaseAlgorithm


class MultinomialNaiveBayes(BaseAlgorithm):

    def __init__(self):
        super().__init__("Multinomial Naive Bayes", ["count_vec_data"])

    def construct_classifier(self):
        return MultinomialNB(alpha=0.01)
