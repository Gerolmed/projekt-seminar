import pandas as pd

from utils.Algorithm import Algorithm
from utils.Data import PosData
from utils.Result import Result
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class CustomPos(Algorithm):
    def get_name(self) -> str:
        return "Custom POS"

    def get_supported_data_type(self) -> str:
        return "pos_data"

    def execute(self, data: PosData) -> Result:
        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', DecisionTreeClassifier(criterion='entropy'))
        ])
        data_array = pd.array(data.train_data, dtype=object)
        clf.fit(data_array, data.train_labels)

        pred_labels = clf.predict(data.test_data)

        precision = precision_score(data.test_labels, pred_labels, average='macro', zero_division=1)
        recall = recall_score(data.test_labels, pred_labels, average='macro', zero_division=1)
        f1 = f1_score(data.test_labels, pred_labels, average='macro', zero_division=1)
        conf_matrix = confusion_matrix(data.test_labels, pred_labels)

        return Result(precision, recall, f1, conf_matrix)