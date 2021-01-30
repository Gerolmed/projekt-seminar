from typing import List

from loading.LoadingUtils import LoadedData
from utils.Data import Data


class DataProvider:

    def execute(self, rawData: LoadedData, test_ids: List[str]) -> Data:
        """Prepares a specific form of training data"""
        raise Exception("Not implemented")
        pass
