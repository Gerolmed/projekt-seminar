from utils. DataProvider import DataProvider
from utils.Data import Data
from loading.LoadingUtils import LoadedData


class GloVe(DataProvider):

    def execute(self, data: Data, rawData: LoadedData) -> Data:
        """Vectorizes the Data via Global Vectorization(GloVe)"""

