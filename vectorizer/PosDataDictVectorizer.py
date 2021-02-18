from sklearn.feature_extraction import DictVectorizer

from utils.Data import Data, DictVecFeatureSetData, FeatureSetData
from utils.Vectorizer import Vectorizer


class PosDataDictVectorizer(Vectorizer):

    def __init__(self):
        super().__init__("feature_set_data",
                         #"dfData"
                         )

    def vectorize(self, data: FeatureSetData) -> DictVecFeatureSetData:
        vectorizer = DictVectorizer(sparse=False)
        x_train = vectorizer.fit_transform(data.train_data)
        x_test = vectorizer.transform(data.test_data)

        return DictVecFeatureSetData(x_train, data.train_labels, x_test, data.test_labels, data.stoppedTestLabels)
