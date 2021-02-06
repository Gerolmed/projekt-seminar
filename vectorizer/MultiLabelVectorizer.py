from sklearn.preprocessing import MultiLabelBinarizer
from utils.Data import MultiLabelData, MultiLabelBinData
from utils.Vectorizer import Vectorizer


class MultiLabelVectorizer(Vectorizer):

    def __init__(self):
        super().__init__("multi_label_data")

    def vectorize(self, data: MultiLabelData) -> MultiLabelBinData:
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(data.train_labels)
        y_test = mlb.transform(data.test_labels)

        return MultiLabelBinData(data.train_data, y_train, data.test_data, y_test)

