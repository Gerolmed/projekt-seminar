from random import random
from typing import List

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from loading.LoadingUtils import LoadedData, seed
from utils.Data import Data
from utils.DataProvider import DataProvider


class CountVect(DataProvider):

    def execute(self, data: Data, rawData: LoadedData) -> Data:

        random.seed(seed)
        random.shuffle(rawData)

        random.seed(seed)
        random.shuffle(full_labels)

        split_parameter = round(len(full_labels) * 0.7)

        sentimentTokens: List[str] = []
        for dataKey, dataValue in trainData.items():
            tokens: List[str] = dataValue.get("tokens")
            for reviewKey, dataData in dataValue.items():
                if reviewKey == "tokens":
                    continue
                labels = dataData["sentiments"]  # todo add _uncertainty as well
                for index, token in enumerate(tokens):
                    if labels[index].endswith("S"):
                        sentimentTokens.append(token)
        print(sentimentTokens)
        vectorizer = CountVectorizer()
        vectorizer.fit(sentimentTokens)
