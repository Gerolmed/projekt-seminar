from typing import List, Dict

from loading.LoadingUtils import LoadedData
from utils.Data import Data, CountVecInputData
from utils.DataProvider import DataProvider


class CountVecDataPreparation(DataProvider):

    @staticmethod
    def vocabulary_append(token: str, vocabulary: dict):
        if token in vocabulary:
            vocabulary[token] += 1
        else:
            vocabulary[token] = 1
        return vocabulary

    def execute(self, rawData: LoadedData, test_ids: List[str]) -> CountVecInputData:
        split_parameter = round(len(rawData) * 0.7)
        all_tokens = []
        all_labels = []

        for dataKey, dataValue in rawData.items():
            tokens: List[str] = dataValue.get("tokens")
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData["sentiments"]
                for index, token in enumerate(tokens):
                    all_tokens.append(token)
                    all_labels.append(labels[index])

        vocabulary: Dict[str, int] = {}

        for dataKey, dataValue in all_tokens[split_parameter:]:
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

        return CountVecInputData(all_tokens[split_parameter:], all_labels[split_parameter:],
                                 all_tokens[:split_parameter], all_labels[:split_parameter], sorted_vocabulary)
