from typing import List, Dict, Tuple
import pandas as pd
from loading.LoadingUtils import LoadedData
from loading.Preprocessing import isStopWord
from utils.Data import CountVecInputData
from utils.DataProvider import DataProvider
import copy

from utils.DataSelector import DataSelector


class CountVecDataPreparation(DataProvider):

    def execute(self, rawData: LoadedData, test_ids: List[str], data_selector: DataSelector) -> CountVecInputData:

        rawData = copy.deepcopy(rawData)

        tagged_sentences: List[Tuple[str, List[Tuple[str, str]]]] = list()
        stoppedTestLabels = dict()
        stopWordLabel = "O"

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
                labels = dataData[data_selector.type_name]  # todo add _uncertainty as well
                sentence = list()
                for index, token in enumerate(tokens):
                    sentence.append((token, data_selector.type_symbol if labels[index].endswith(data_selector.type_symbol) else "O"))
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
                    if isStopWord(token):
                        stoppedTestLabels.setdefault(len(y_test)-1, stopWordLabel)
                else:
                    x_train.append(token)
                    y_train.append(label)
                    if label.endswith(data_selector.type_symbol):
                        vocabulary[token] += 1

        sorted_vocabulary = {}
        sorted_keys = sorted(vocabulary, key=vocabulary.get, reverse=True)
        for w in sorted_keys:
            sorted_vocabulary[w] = vocabulary[w]

        index_vocabulary = {}
        for index, key in enumerate(sorted_vocabulary):
            index_vocabulary[key] = index

        """if data_selector.type_symbol == "S":
            dfs = pd.DataFrame({"Tokens": sorted_vocabulary.keys(), "Count": sorted_vocabulary.values()})
            dfi = pd.DataFrame({"Tokens": index_vocabulary.keys(), "Index": index_vocabulary.values()})
            dfs.to_csv("WordCount.csv", index=False)
            dfi.to_csv("WordIndex.csv", index=False)"""

        return CountVecInputData(x_train, y_train, x_test, y_test, index_vocabulary, stoppedTestLabels)
