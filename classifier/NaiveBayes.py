from sklearn.naive_bayes import MultinomialNB
from classifier.BaseClassifier import BaseClassifier


class MultinomialNaiveBayes(BaseClassifier):

    def __init__(self):
        super().__init__("Multinomial Naive Bayes", ["count_vec_data",
                                                     "dict_vec_feature_set_data"])

    def construct_classifier(self):
        return MultinomialNB(alpha=0.01)
