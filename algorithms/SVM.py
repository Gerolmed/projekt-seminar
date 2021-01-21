import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils.Algorithm import Algorithm
from utils.Data import PosData
from utils.Result import Result


class SupportVectorMachine(Algorithm):

    def get_name(self) -> str:
        return "Support Vector Machine (SVM)"

    def get_supported_data_type(self) -> str:
        return "pos_data"

    def execute(self, data: PosData) -> Result:

        clf = Pipeline([
            ('standardscaler', StandardScaler()),
            ('linearsvc', LinearSVC(random_state=0, tol=1e-05))
        ])
        data_array = pd.array(data.train_data, dtype=object)
        clf.fit(data_array, data.train_labels)

        pred_labels = clf.predict(data.test_data)

        precision = precision_score(data.test_labels, pred_labels, average="macro")
        recall = recall_score(data.test_labels, pred_labels, average="macro")
        f1 = f1_score(data.test_labels, pred_labels, average="macro")
        conf_matrix = confusion_matrix(data.test_labels, pred_labels)

        return Result(precision, recall, f1, conf_matrix)
