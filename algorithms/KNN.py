from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import pandas as pd

from utils.Algorithm import Algorithm
from utils.Data import PosData
from utils.Result import Result


class KNearestNeighbor(Algorithm):

    def get_name(self) -> str:
        return "K-Nearest Neighbor"

    def get_supported_data_type(self) -> str:
        return "pos_data"

    def execute(self, data: PosData) -> Result:
        data.train_data = data.train_data[:round(len(data.train_data)/10)]
        data.train_labels = data.train_labels[:round(len(data.train_labels)/10)]
        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', KNeighborsClassifier(n_neighbors=3))
        ])

        data_array = pd.array(data.train_data)

        clf.fit(data_array, data.train_labels)

        pred_labels = clf.predict(data.test_data)

        precision = precision_score(data.test_labels, pred_labels, average="macro")
        recall = recall_score(data.test_labels, pred_labels, average="macro")
        f1 = f1_score(data.test_labels, pred_labels, average="macro")
        conf_matrix = confusion_matrix(data.test_labels, pred_labels)

        return Result(precision, recall, f1, conf_matrix)