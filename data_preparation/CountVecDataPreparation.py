from typing import List, Dict

from loading.LoadingUtils import LoadedData
from utils.Data import CountVecInputData
from utils.DataProvider import DataProvider
import copy


class CountVecDataPreparation(DataProvider):

    @staticmethod
    def vocabulary_append(token: str, vocabulary: dict):
        if token in vocabulary:
            vocabulary[token] += 1
        else:
            vocabulary[token] = 1
        return vocabulary

    def execute(self, rawData: LoadedData, test_ids: List[str]) -> CountVecInputData:
        raw_data = copy.deepcopy(rawData)
        split_parameter = round(len(raw_data) * 0.7)
        all_tokens = []
        all_labels = []

        for dataKey, dataValue in raw_data.items():
            tokens: List[str] = dataValue.get("tokens")
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData["sentiments"]
                for index, token in enumerate(tokens):
                    all_tokens.append(token)
                    all_labels.append(labels[index])

        vocabulary: Dict[str, int] = {}

        for i, (dataKey, dataValue) in enumerate(raw_data.items()):
            if i > len(all_tokens[split_parameter:]):
                break
            tokens: List[str] = dataValue.get("tokens")
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData["sentiments"]
                for index, token in enumerate(tokens):
                    if labels[index].endswith("S"):
                        vocabulary = self.vocabulary_append(token, vocabulary)

        sorted_vocabulary = {}
        sorted_keys = sorted(vocabulary, key=vocabulary.get, reverse=True)
        for w in sorted_keys:
            sorted_vocabulary[w] = vocabulary[w]
        index_vocabulary = {}
        for index, key in enumerate(sorted_vocabulary):
            index_vocabulary[key] = index

        return CountVecInputData(all_tokens[split_parameter:], all_labels[split_parameter:],
                                 all_tokens[:split_parameter], all_labels[:split_parameter], index_vocabulary)
