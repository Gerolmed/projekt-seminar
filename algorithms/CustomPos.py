import pandas as pd

from utils.Algorithm import Algorithm
from utils.Data import PosData
from utils.Result import Result
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


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
        print("accuracy", clf.score(data.test_data, data.test_labels))

        return Result(None, None, None, None)
