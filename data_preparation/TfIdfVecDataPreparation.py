import copy
import math
from collections import Counter
from typing import List, Dict, Tuple

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from loading.LoadingUtils import LoadedData
from utils.Data import CountVecInputData
from utils.DataProvider import DataProvider


class CountVecDataPreparation(DataProvider):
    def execute(self, rawData: LoadedData, test_ids: List[str]) -> TfIdfVecInputData:

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
                    sentence.append((token, labels[index]))
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

        return CountVecInputData(x_train, y_train, x_test, y_test, index_vocabulary)


def transform(dataset, vocab):
    row = []
    col = []
    values = []
    for ibx, document in enumerate(dataset):
        word_freq = dict(Counter(document.split()))
        for word, freq in word_freq.items():
            col_index = vocab.get(word, -1)
            if col_index != -1:
                if len(word) < 2:
                    continue
                col.append(col_index)
                row.append(ibx)
                td = freq / float(len(document))  # the number of times a word occured in a document
                idf_ = 1 + math.log((1 + len(dataset)) / float(1 + idf(word)))
                values.append((td) * (idf_))
        return normalize(csr_matrix(((values), (row, col)), shape=(len(dataset), len(vocab))), norm='l2')
