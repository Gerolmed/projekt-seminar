from sklearn.feature_extraction.text import TfidfVectorizer

from utils.Data import CountVecInputData, TfIdfVecData
from utils.Vectorizer import Vectorizer


class TfIdfVec(Vectorizer):

    def __init__(self):
        super().__init__("count_vec_input_data")

    def vectorize(self, data: CountVecInputData) -> TfIdfVecData:
        vectorizer = TfidfVectorizer(analyzer='word', vocabulary=data.vocabulary)
        x_train = vectorizer.fit_transform(data.train_data)
        x_test = vectorizer.transform(data.test_data)

        return TfIdfVecData(x_train, data.train_labels, x_test, data.test_labels)
