from algorithms.BaseAlgorithm import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier


class DecisionTrees(BaseAlgorithm):

    def __init__(self):
        super().__init__("Decision Trees", ["dict_vec_pos_data"])

    def construct_classifier(self):
        return DecisionTreeClassifier(criterion='entropy')
