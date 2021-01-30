from utils.Data import Data


class Vectorizer:
    def __init__(self, supported_data_type: str):
        self.supported_data_type = supported_data_type

    def vectorize(self, data: Data) -> Data:
        """Returns vectorized data object"""
        pass

    def get_supported_data_type(self) -> str:
        """Returns the supported data type id"""
        return self.supported_data_type
