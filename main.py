from typing import List, Dict

from algorithms.DecisionTrees import DecisionTrees
from algorithms.NaiveBayes import MultinomialNaiveBayes
from algorithms.KNN import KNearestNeighbor
from algorithms.SVM import SupportVectorMachine
from data_preparation.ComparisonDataPreparation import ComparisonDataPreparation
from data_preparation.PosPreparation import PosPreparation
from loading.LoadingUtils import LoadingUtils
from utils.Algorithm import Algorithm
from utils.Data import Data
from utils.DataProvider import DataProvider
from utils.Result import Result
from utils.Vectorizer import Vectorizer
from vectorizer.CountVectorizer import CountVec
from vectorizer.PosDataDictVectorizer import PosDataDictVectorizer

data_providers: List[DataProvider] = [
    ComparisonDataPreparation(),
    PosPreparation()
]
"""Prepares data for vectorizer (or directly for algorithm)"""

vectorizers: List[Vectorizer] = [
    PosDataDictVectorizer(),
    CountVec()
]
"""The vectorizers to add vectorized data based on prepared data"""


algorithms: List[Algorithm] = [
    MultinomialNaiveBayes(),
    # SupportVectorMachine(),
    # DecisionTrees(),
    # KNearestNeighbor(),
]
"""The Algorithms to use"""

data_dict: Dict[str, Data] = dict()
"""Dictionary to store different data types by id"""

# Load raw and basic data
[rawData, test_ids] = LoadingUtils.read_data(filename=r'./data_laptop_absa.json', testIdsFile=r"./test_ids.json")
# Create different base data formats
for data_provider in data_providers:
    data = data_provider.execute(rawData, test_ids)
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
        print(f"Executing {algorithm.get_name()} with {selected_data.data_type}...")
        result = algorithm.execute(selected_data)
        print(f"Finished executing {algorithm.get_name()} in {round((result.train_time + result.test_time), 3)} "
              f"sec ({round((result.train_time + result.test_time)/60, 3)} min)!")

        results.append(result)

# Prints all results in a fixed format after execution
for result in results:
    print("============================================")
    print(f"Name: {result.algorithm_name}")
    print(f"Data: {result.data_name}")
    print("")
    print(f"Train Time: {result.train_time} sec")
    print(f"Test Time: {result.test_time} sec")
    print("")
    print(f"Precision: {result.precision}")
    print(f"Recall: {result.recall}")
    print(f"F1-measure: {result.f1}")
    print("")
    print(f"Confusion Matrix\n{result.confusion_matrix}")
    print("============================================")
    print("")
