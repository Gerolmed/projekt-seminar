from utils.Algorithm import Algorithm
from utils.Data import TfidfVectorizerData
from utils.Result import Result
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class NaiveBayes(Algorithm):

    def get_name(self) -> str:
        return "Naive Bayes"

    def get_supported_data_type(self) -> str:
        return "tfidf_vectorized"

    def execute(self, data: TfidfVectorizerData) -> Result:

        model = MultinomialNB(alpha=0.01)
        model.fit(data.train_data, data.train_labels)

        labels_predicted = model.predict(data.test_data)

        conf_matrix = confusion_matrix(data.test_labels, labels_predicted)
        precision = precision_score(data.test_labels, labels_predicted, average="macro")
        recall = recall_score(data.test_labels, labels_predicted, average="macro")
        f1 = f1_score(data.test_labels, labels_predicted, average="macro")

        return Result(precision, recall, f1, conf_matrix)
