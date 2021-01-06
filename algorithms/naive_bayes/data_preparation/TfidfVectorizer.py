from loading.LoadingUtils import LoadedData
from utils.DataProvider import DataProvider
from utils.Data import Data
from loading import LoadingUtils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd


class TfidfVectorizer(DataProvider):

    def execute(self, data: Data, rawData: LoadedData) -> Data:
        return Data("TfidfVectorizer")