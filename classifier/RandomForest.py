from classifier.BaseClassifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier


class RandomForest(BaseClassifier):

    def __init__(self):
        super().__init__("RandomForestClassifier", ["dict_vec_feature_set_data"])

    def construct_classifier(self):
        return RandomForestClassifier(criterion='entropy')
