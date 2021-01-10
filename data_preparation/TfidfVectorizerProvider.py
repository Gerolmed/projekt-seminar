import random
from typing import List

from loading.LoadingUtils import LoadedData, seed
from utils.DataProvider import DataProvider
from utils.Data import Data, BasicData, TfidfVectorizerData
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerProvider(DataProvider):

    def execute(self, data: BasicData, rawData: LoadedData) -> Data:
        vocabulary = list()

        # Collect vocabulary
        for key, value in rawData.items():
            tokens: List[str] = value.get("tokens")
            vocabulary.extend(tokens)

        vocabulary = set(vocabulary)

        # Instantiate vectorizer
        td = TfidfVectorizer(vocabulary)

        # Collect data
        # Collect vocabulary
        full_data: List[List[int]] = list()
        full_labels: List[str] = list()

        for dataKey, dataValue in rawData.items():
            tokens: List[str] = dataValue.get("tokens")
            transformed_tokens: List[int] = td.transform(" ".join(tokens)).toArray()[0]
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                # TODO: add option to also consider uncertain sentiments
                sentiment_indices = [i for i, x in enumerate(dataData["sentiments"]) if x == "whatever"]
                for index, token in enumerate(tokens):
                    full_data.append(transformed_tokens)
                    full_labels.append("sentiment" if index in sentiment_indices else "none")
        random.seed(seed)
        random.shuffle(full_data)

        random.seed(seed)
        random.shuffle(full_labels)

        split_parameter = round(len(full_labels) * 0.7)

        return TfidfVectorizerData(full_data[split_parameter:], full_labels[split_parameter:],
                                   full_data[:split_parameter], full_labels[:split_parameter])
