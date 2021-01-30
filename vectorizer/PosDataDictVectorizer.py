from sklearn.feature_extraction import DictVectorizer

from utils.Data import PosData, DictVecPosData
from utils.Vectorizer import Vectorizer


class PosDataDictVectorizer(Vectorizer):

    def __init__(self):
        super().__init__("pos_data")

    def vectorize(self, data: PosData) -> DictVecPosData:
        vectorizer = DictVectorizer(sparse=False)
        x_train = vectorizer.fit_transform(data.train_data)
        x_test = vectorizer.transform(data.test_data)

        return DictVecPosData(x_train, data.train_labels, x_test, data.test_labels)
