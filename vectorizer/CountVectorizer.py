from sklearn.feature_extraction.text import CountVectorizer

from utils.Data import CountVecInputData, CountVecData
from utils.Vectorizer import Vectorizer


class CountVec(Vectorizer):

    def __init__(self):
        super().__init__("count_vec_input_data")

    def vectorize(self, data: CountVecInputData) -> CountVecData:
        vectorizer = CountVectorizer(sparse=False, vocabulary=data.vocabulary)
        x_train = vectorizer.fit_transform(data.train_data)
        x_test = vectorizer.transform(data.test_data)

        return CountVecData(x_train, data.train_labels, x_test, data.test_labels)
