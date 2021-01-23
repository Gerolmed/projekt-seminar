from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, CategoricalNB, BernoulliNB
from sklearn.pipeline import Pipeline
import pandas as pd

from utils.Algorithm import Algorithm
from utils.Data import PosData
from utils.Result import Result


class NaiveBayes(Algorithm):

    def get_name(self) -> str:
        return "Naive Bayes"

    def get_supported_data_type(self) -> str:
        return "pos_data"

    def execute(self, data: PosData) -> Result:

        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', MultinomialNB(alpha=0.01))
        ])

        data_array = pd.array(data.train_data)

        clf.fit(data_array, data.train_labels)

        pred_labels = clf.predict(data.test_data)

        precision = precision_score(data.test_labels, pred_labels, average="macro")
        recall = recall_score(data.test_labels, pred_labels, average="macro")
        f1 = f1_score(data.test_labels, pred_labels, average="macro")
        conf_matrix = confusion_matrix(data.test_labels, pred_labels)

        return Result(precision, recall, f1, conf_matrix)