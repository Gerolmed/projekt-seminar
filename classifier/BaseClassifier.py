from typing import List, Any

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils.Classifier import Classifier
from utils.Data import FeatureSetData
from utils.Delta import delta
from utils.Result import Result
import pandas as pd


class BaseClassifier(Classifier):
    def __init__(self, name: str, supported_types: List[str]):
        self.name = name
        self.supported_types = supported_types

    def get_name(self) -> str:
        return self.name

    def get_supported_data_types(self) -> List[str]:
        return self.supported_types

    def construct_classifier(self) -> Any:
        """Creates the used classifier. Must be overridden!"""
        pass

    def execute(self, data: FeatureSetData) -> Result:
        clf = self.construct_classifier()
        [_, train_delta] = self.train(clf, data.train_data, data.train_labels)
        [pred_labels, test_delta] = self.test(clf, data.test_data)

        for index, label in data.stoppedTestLabels.items():
            pred_labels[index] = label

        precision = precision_score(data.test_labels, pred_labels, average='macro')
        recall = recall_score(data.test_labels, pred_labels, average='macro')
        f1 = f1_score(data.test_labels, pred_labels, average='macro')
        conf_matrix = confusion_matrix(data.test_labels, pred_labels)

        return Result(precision, recall, f1, conf_matrix, self.get_name(), data.data_type, round(train_delta, 3),
                      round(test_delta, 3))

    @delta
    def train(self, classifier, X: List, Y: List[str]):
        classifier.fit(X, Y)

    @delta
    def test(self, classifier, X: List):
        return classifier.predict(X)
