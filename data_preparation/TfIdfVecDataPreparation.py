import copy
import math
from collections import Counter
from typing import List, Dict, Tuple

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from loading.LoadingUtils import LoadedData
from utils.Data import CountVecInputData, TfIdfVecInputData
from utils.DataProvider import DataProvider


class TfIdfVecInputDataPreparation(DataProvider):
    def execute(self, rawData: LoadedData, test_ids: List[str]) -> TfIdfVecInputData:

        rawData = copy.deepcopy(rawData)

        tagged_sentences: List[Tuple[str, List[Tuple[str, str]]]] = list()

        x_train: List[str] = []
        y_train: List[str] = []

        x_test: List[str] = []
        y_test: List[str] = []

        token_frequency: Dict[str, float] = {}
        document_count: int = 0
        vocabulary: Dict[str, int] = {}
        token_frequency_per_doc: Dict[str, Dict[str, int]] = dict()
        document_frequency: Dict[str, float] = dict()
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
                usedTokens: List[str] = []

                for index, token in enumerate(tokens):
                    label = labels[index]
                    if dataKey in test_ids:
                        x_test.append(token)
                        y_test.append(label)
                    else:
                        vocabulary[token] = vocabulary.get(token, 0)
                        if label.endswith("S"):
                            vocabulary[token] += 1
                            if token not in usedTokens:
                                document_frequency[token] = document_frequency.get(token, 0) + 1
                                usedTokens.append(token)
                            token_frequencies = token_frequency_per_doc.get(dataKey, dict())
                            token_frequencies[token] = token_frequencies.get(token, 0) + 1
                            token_frequency_per_doc[dataKey] = token_frequencies

                        x_train.append(token)
                        y_train.append(label)
                break  # currently just using first review

        # Sort vocabulary
        sorted_vocabulary = {}
        sorted_keys = sorted(vocabulary, key=vocabulary.get, reverse=True)
        for w in sorted_keys:
            sorted_vocabulary[w] = vocabulary[w]

        index_vocabulary: List[str] = list()
        for index, key in enumerate(sorted_vocabulary):
            index_vocabulary.append(key)

        # Calculate idf
        for document, frequency in document_frequency.items():
            document_frequency[document] = 1 + math.log((1 + document_count) / float(1 + frequency))

        # Produce matrix values
        row: List[int] = []
        col: List[int] = []
        values: List[float] = []

        for document_index, (document, frequencies) in enumerate(token_frequency_per_doc.items()):
            for token, frequency in frequencies.items():
                row.append(document_index)
                col.append(index_vocabulary.index(token))
                values.append(frequency * document_frequency.get(token, 0))
        print("")
        return TfIdfVecInputData(x_train, y_train, x_test, y_test,
                                 normalize(
                                     csr_matrix((values, (row, col)), shape=(document_count, len(vocabulary))),
                                     norm='l2'
                                 )
                                 )
