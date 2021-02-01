from typing import Dict

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from utils.Data import TfIdfVecInputData, TfIdfVecData
from utils.Vectorizer import Vectorizer


def get_tdidf(token: str, token_frequency: Dict[str, float], document_frequency: Dict[str, float]):
    return token_frequency.get(token, 0) * document_frequency.get(token, 0)


class TfIdfVec(Vectorizer):

    def __init__(self):
        super().__init__("tfidf_vec_input_data")

    def vectorize(self, data: TfIdfVecInputData) -> TfIdfVecData:



        return TfIdfVecData(x_train, data.train_labels, x_test, data.test_labels)
