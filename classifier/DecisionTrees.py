from classifier.BaseClassifier import BaseClassifier
from sklearn.tree import DecisionTreeClassifier


class DecisionTrees(BaseClassifier):

    def __init__(self):
        super().__init__("Decision Trees", ["dict_vec_feature_set_data"])

    def construct_classifier(self):
        return DecisionTreeClassifier(criterion='entropy')
