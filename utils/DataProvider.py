from typing import List

from loading.LoadingUtils import LoadedData
from utils.Data import Data
from utils.DataSelector import DataSelector


class DataProvider:

    def execute(self, rawData: LoadedData, test_ids: List[str], data_selector: List[DataSelector]) -> Data:
        """Prepares a specific form of training data"""
        raise Exception("Not implemented")
        pass
