from utils.Algorithm import Algorithm
from utils.Data import Data
from utils.Result import Result
from loading.LoadingUtils import LoadingUtils
import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class NaiveBayes(Algorithm):

    def get_name(self) -> str:
        return "Naive Bayes"

    def execute(self, data: Data) -> Result:

        train_token_array = []
        for token in data.train_tokens:
            train_token_array.append(token)

        model = CategoricalNB()
        model.fit(train_token_array, data.train_labels)

        labels_predicted = model.predict(data.test_tokens)

        conf_matrix = confusion_matrix(data.test_labels, labels_predicted)
        precision = precision_score(data.test_labels, labels_predicted)
        recall = recall_score(data.test_labels, labels_predicted)
        f1 = f1_score(data.test_labels, labels_predicted)

        return Result(precision, recall, f1, conf_matrix)
