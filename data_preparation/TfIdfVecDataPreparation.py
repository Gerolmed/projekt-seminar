import copy
import math
from collections import Counter
from typing import List, Dict, Tuple

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from loading.LoadingUtils import LoadedData
from utils.Data import CountVecInputData, TfIdfVecInputData
from utils.DataProvider import DataProvider


class CountVecDataPreparation(DataProvider):
    def execute(self, rawData: LoadedData, test_ids: List[str]) -> TfIdfVecInputData:

        rawData = copy.deepcopy(rawData)

        tagged_sentences: List[Tuple[str, List[Tuple[str, str]]]] = list()

        x_train: List[str] = []
        y_train: List[str] = []

        x_test: List[str] = []
        y_test: List[str] = []

        token_frequency: Dict[str, float] = {}
        document_count: int = 0

        for dataKey, dataValue in rawData.items():
            tokens: List[str] = dataValue.get("tokens")

            # Increase document count if train document
            if dataKey not in test_ids:
                document_count += 1

            # Collect tokens
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue

                labels = dataData["sentiments"]  # todo add _uncertainty as well
                sentence = list()
                for index, token in enumerate(tokens):
                    sentence.append((token, labels[index]))
                tagged_sentences.append((dataKey, sentence))

        document_frequency: Dict[str, float] = dict()

        # Collect data and raw frequencies
        for group in tagged_sentences:
            document_id: str = group[0]
            tagged = group[1]
            used_tokens_for_document_frequency: List[str] = []

            for index in range(len(tagged)):

                token = tagged[index][0]
                label = tagged[index][1]

                if document_id in test_ids:
                    x_test.append(token)
                    y_test.append(label)
                else:
                    x_train.append(token)
                    y_train.append(label)
                    if label.endswith("S"):
                        token_frequency[token] = token_frequency.get(token, 0) + 1

                        # If not already used for document frequency in this document do so
                        if token not in used_tokens_for_document_frequency:
                            used_tokens_for_document_frequency.append(token)
                            document_frequency[token] = document_frequency.get(token, 0) + 1
                    else:
                        token_frequency[token] = token_frequency.get(token, 0)

        for token, frequency in token_frequency:
            token_frequency[token] = float(frequency)/document_count
        for token, frequency in document_frequency:
            token_frequency[token] = 1 + math.log((1 + document_count) / float(1 + frequency))
        return TfIdfVecInputData(x_train, y_train, x_test, y_test, token_frequency, document_frequency)


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
