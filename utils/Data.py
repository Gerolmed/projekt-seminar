from typing import List, Union, Dict

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
