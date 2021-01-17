from utils. DataProvider import DataProvider
from utils.Data import Data
from loading.LoadingUtils import LoadedData
from typing import List
from data_preparation.StopWords import removeStopWords


class GloVe(DataProvider):

    def execute(self, data: Data, rawData: LoadedData) -> Data:
        """Vectorizes the Data via Global Vectorization(GloVe)"""

        preparedData = removeStopWords(rawData)
        vocabulary = list()

        # Collect vocabulary
        for key, value in rawData.items():
            tokens: List[str] = value.get("tokens")
            vocabulary.extend(tokens)
        # TODO implementation
        return data
