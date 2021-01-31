from typing import List, Union, Dict, Any

TrainTokens = List[Union[Dict[str, List[str]], List[str]]]
TrainLabels = List[List[str]]
TestTokens = List[Union[Dict[str, List[str]], List[str]]]
TestLabels = List[List[str]]


class Data:
    """The data split into test and training"""

    def __init__(self, data_type: str):
        self.data_type = data_type


class BasicData(Data):
    def __init__(self, train_tokens: TrainTokens, train_labels: TrainLabels, test_tokens: TestTokens,
                 test_labels: TestLabels, n_tags, labelclass_to_id):
        super().__init__("basic")
        self.train_tokens = train_tokens
        self.train_labels = train_labels
        self.test_tokens = test_tokens
        self.test_labels = test_labels
        self.labelclass_to_id = labelclass_to_id
        self.n_tags = n_tags


class TfidfVectorizerData(Data):
    def __init__(self, train_data: List[List[int]], train_labels: List[str], test_data: List[List[int]],
                 test_labels: List[str]):
        super().__init__("tfidf_vectorized")
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


class PosData(Data):
    def __init__(self, train_data: List[Dict[str, bool]], train_labels: List[str], test_data: List[Dict[str, bool]],
                 test_labels: List[str]):
        super().__init__("pos_data")
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


class DictVecPosData(Data):
    def __init__(self, train_data: List[Dict[str, Any]], train_labels: List[str], test_data: List[Dict[str, Any]],
                 test_labels: List[str]):
        super().__init__("dict_vec_pos_data")
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


class CountVecInputData(Data):
    def __init__(self, train_data: List[str], train_labels: List[str], test_data: List[str],
                 test_labels: List[str], vocabulary: Dict[str, int]):
        super().__init__("count_vec_input_data")
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.vocabulary = vocabulary


class CountVecData(Data):
    def __init__(self, train_data: List[Dict[str, Any]], train_labels: List[str], test_data: List[Dict[str, Any]],
                 test_labels: List[str]):
        super().__init__("count_vec_data")
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


class TfIdfVecData(Data):
    def __init__(self, train_data: List[Dict[str, Any]], train_labels: List[str], test_data: List[Dict[str, Any]],
                 test_labels: List[str]):
        super().__init__("tfidf_vec_data")
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
