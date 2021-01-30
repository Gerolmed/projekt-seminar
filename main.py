from datetime import time
from typing import List, Dict
from algorithms.SVM import SupportVectorMachine
from algorithms.DecisionTrees import DecisionTrees
from algorithms.NaiveBayes import NaiveBayes
from algorithms.Type1 import Type1
from algorithms.KNN import KNearestNeighbor
from data_preparation.GloVe import GloVe
from data_preparation.PosPreparation import PosPreparation
from data_preparation.TfidfVectorizerProvider import TfidfVectorizerProvider
from loading.LoadingUtils import LoadingUtils
from utils.Algorithm import Algorithm
from utils.Data import Data
from utils.DataProvider import DataProvider
from utils.Result import Result
from utils.Vectorizer import Vectorizer
from vectorizer.PosDataDictVectorizer import PosDataDictVectorizer

algorithms: List[Algorithm] = [  # Type1(),
    # SupportVectorMachine(),
    NaiveBayes(),
    # DecisionTrees(),
    KNearestNeighbor()
]
data_providers: List[DataProvider] = [TfidfVectorizerProvider(), GloVe(), PosPreparation()]
vectorizers: List[Vectorizer] = [PosDataDictVectorizer()]


data_dict: Dict[str, Data]= dict()
"""Dictionary to store different data types by id"""

# Load raw and basic data
[basicData, rawData, test_ids] = LoadingUtils.read_data(filename=r'./data_laptop_absa.json',
                                                        testIdsFile=r"./test_ids.json")


data_dict.setdefault(basicData.data_type, basicData)

# Create different base data formats
for data_provider in data_providers:
    data = data_provider.execute(basicData, rawData, test_ids)
    data_dict.setdefault(data.data_type, data)

# Vectorize data and add it to data list
for vectorizer in vectorizers:
    selected_data = data_dict.get(vectorizer.get_supported_data_type())
    data = vectorizer.vectorize(selected_data)
    data_dict.setdefault(data.data_type, data)

results: List[Result] = []
"""Collects the results of the different algorithms for further usages"""

# Run and execute every algorithm
for algorithm in algorithms:
    for data_type in algorithm.get_supported_data_types():
        selected_data = data_dict.get(data_type)
        print("Executing " + algorithm.get_name() + " with " + selected_data.data_type + "...")
        result = algorithm.execute(selected_data)
        print("Finished executing " + algorithm.get_name() + "!")

        results.append(result)

for result in results:
    print("============================================")
    print("Name: " + str(result.algorithm_name))
    print("Data: " + str(result.data_name))
    print("")
    print("Precision: " + str(result.precision))
    print("Recall: " + str(result.recall))
    print("F1-measure: " + str(result.f1))
    print(f'\n{result.confusion_matrix}\n')
    print("============================================")
    print("")
