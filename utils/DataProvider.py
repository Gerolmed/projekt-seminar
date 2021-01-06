from loading.LoadingUtils import LoadedData
from utils.Data import Data


class DataProvider:

    def execute(self, data: Data, rawData: LoadedData) -> Data:
        """Prepares a specific form of training data"""
        raise Exception("Not implemented")
        pass
