from utils.TrainingData import TrainingData
from utils.Data import Data
from loading import LoadingUtils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd


class TfidfVectorizer(TrainingData):

    def get_name(self) -> str:
        return "TfidfVectorizer"

    def convert_data_to_array(self, data: Data):
        pass
