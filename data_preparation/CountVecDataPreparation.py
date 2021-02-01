from typing import List, Dict, Tuple

from loading.LoadingUtils import LoadedData
from utils.Data import CountVecInputData
from utils.DataProvider import DataProvider
import copy


class CountVecDataPreparation(DataProvider):
    # TODO change TestData to TestIDs
    def execute(self, rawData: LoadedData, test_ids: List[str]) -> CountVecInputData:

        rawData = copy.deepcopy(rawData)

        tagged_sentences: List[Tuple[str, List[Tuple[str, str]]]] = list()

        x_train = []
        y_train: List[str] = []

        x_test = []
        y_test: List[str] = []

        vocabulary: Dict[str, int] = {}

        for dataKey, dataValue in rawData.items():
            tokens: List[str] = dataValue.get("tokens")
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData["sentiments"]  # todo add _uncertainty as well
                sentence = list()
                for index, token in enumerate(tokens):
                    sentence.append((token, "S" if labels[index].endswith("S") else "O"))
                    vocabulary.setdefault(token, 0)
                tagged_sentences.append((dataKey, sentence))

        for group in tagged_sentences:
            tagged = group[1]
            for index in range(len(tagged)):

                token = tagged[index][0]
                label = tagged[index][1]

                if group[0] in test_ids:
                    x_test.append(token)
                    y_test.append(label)
                else:
                    x_train.append(token)
                    y_train.append(label)
                    if label.endswith("S"):
                        vocabulary[token] += 1

        sorted_vocabulary = {}
        sorted_keys = sorted(vocabulary, key=vocabulary.get, reverse=True)
        for w in sorted_keys:
            sorted_vocabulary[w] = vocabulary[w]

        index_vocabulary = {}
        for index, key in enumerate(sorted_vocabulary):
            index_vocabulary[key] = index

        # print(y_train)

        return CountVecInputData(x_train, y_train, x_test, y_test, index_vocabulary)
