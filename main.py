from typing import List, Dict
from algorithms.SVM import SupportVectorMachine
from algorithms.CustomPos import CustomPos
from algorithms.NaiveBayes import NaiveBayes
from algorithms.Type1 import Type1
from data_preparation.GloVe import GloVe
from data_preparation.PosPreparation import PosPreparation
from data_preparation.TfidfVectorizerProvider import TfidfVectorizerProvider
from loading.LoadingUtils import LoadingUtils
from utils.Algorithm import Algorithm
from utils.Data import Data
from utils.DataProvider import DataProvider
from utils.Result import Result


algorithms: List[Algorithm] = [SupportVectorMachine(), NaiveBayes()]
data_providers: List[DataProvider] = [TfidfVectorizerProvider(), GloVe(), PosPreparation()]


[basicData, rawData] = LoadingUtils.read_data(filename=r'./data_laptop_absa.json')

data_dict: Dict[str, Data] = dict()

data_dict.setdefault(basicData.data_type, basicData)

for data_provider in data_providers:
    data = data_provider.execute(basicData, rawData)
    data_dict.setdefault(data.data_type, data)

results: Dict[str, Result] = dict()
for algorithm in algorithms:
    selected_data = data_dict.get(algorithm.get_supported_data_type())
    result = algorithm.execute(selected_data)
    results.setdefault(algorithm.get_name(), result)

for key, value in results.items():
    print("============================================")
    print('Name: ' + str(key))
    print('Precision: ' + str(value.precision))
    print('Recall: ' + str(value.recall))
    print('F1-measure: ' + str(value.f1))
    print(f'\n{value.confusion_matrix}\n')
    print("============================================")
    print("")


