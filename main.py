import copy
from typing import List, Dict, Tuple

from algorithms.ComparisonClassifier import ComparisonClassifier
from algorithms.DecisionTrees import DecisionTrees
from algorithms.NaiveBayes import MultinomialNaiveBayes
from algorithms.KNN import KNearestNeighbor
from algorithms.SVM import SupportVectorMachine
from data_preparation.ComparisonDataPreparation import ComparisonDataPreparation
from data_preparation.CountVecDataPreparation import CountVecDataPreparation
from data_preparation.PosPreparation import PosPreparation
from loading.LoadingUtils import LoadingUtils
from utils.Algorithm import Algorithm
from utils.Data import Data
from utils.DataProvider import DataProvider
from utils.DataSelector import DataSelector
from utils.Result import Result
from utils.Vectorizer import Vectorizer
from vectorizer.CountVectorizer import CountVec
from vectorizer.PosDataDictVectorizer import PosDataDictVectorizer

data_selectors: List[DataSelector] = [
    DataSelector("sentiments", "S"),
    DataSelector("aspects", "A")
]

data_providers: List[DataProvider] = [
    ComparisonDataPreparation(),
    # PosPreparation(),
    CountVecDataPreparation(),
]
"""Prepares data for vectorizer (or directly for algorithm)"""

vectorizers: List[Vectorizer] = [
    # PosDataDictVectorizer(),
    CountVec(),
]
"""The vectorizers to add vectorized data based on prepared data"""

algorithms: List[Algorithm] = [
    ComparisonClassifier(),
    # MultinomialNaiveBayes(),
    SupportVectorMachine(),
    # DecisionTrees()
    # KNearestNeighbor(),
]
"""The Algorithms to use"""


def run_pipeline(data_selector: DataSelector):
    data_dict: Dict[str, Data] = dict()
    """Dictionary to store different data types by id"""

    # Load raw and basic data
    [raw_data, test_ids] = LoadingUtils.read_data(r'./data_laptop_absa.json', r"./test_ids.json", data_selector)
    # Create different base data formats
    for data_provider in data_providers:
        data = data_provider.execute(raw_data, test_ids, data_selector)
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

            if selected_data is None:
                print(f"Skipping type {data_type} for {algorithm.get_name()} because data is missing!")
                continue

            print(f"Executing {algorithm.get_name()} with {selected_data.data_type} ({data_selector.type_name})...")
            result = algorithm.execute(selected_data)
            print(f"Finished executing {algorithm.get_name()} in {round((result.train_time + result.test_time), 3)} "
                  f"sec ({round((result.train_time + result.test_time) / 60, 3)} min)!")

            results.append(result)
    return results


def print_results(data_selector: DataSelector, results: List[Result]):
    print("============================================")
    print(f"Printing results for {data_selector.type_name}")
    print("============================================")

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


# #################################
# Result combination
# #################################

def combine_results(total_results: Dict[DataSelector, List[Result]], index=0, previous: List[List[Tuple[DataSelector, Result]]] = []) -> List[List[Tuple[DataSelector, Result]]]:

    if len(total_results.items()) <= index:
        return previous

    out_list: List[List[Tuple[DataSelector, Result]]] = []
    [selector, results] = list(total_results.items())[index]

    if index == 0:
        for result in results:
            out_list.append([(selector, result)])
    else:
        for combination in previous:
            for result in results:
                new_list = combination.copy()
                new_list.append((selector, result))
                out_list.append(new_list)
    return combine_results(total_results, index + 1, out_list)


# #################################
# Main Run
# #################################

def print_combination(combination: List[Tuple[DataSelector, Result]], f1_score: float):
    used_text: str = "Used: "
    calc_text: str = "("
    for selector, result in combination:
        used_text += f"{selector.type_name} ({result.data_name} -> {result.algorithm_name}) "
        calc_text += f"{str(result.f1)} + "
    calc_text = calc_text[:len(calc_text) - 3]
    calc_text += f") / {str(len(combination))}"
    calc_text += " = " + str(f1_score)

    print("============================================")
    print(used_text)
    print(calc_text)
    print(f"F1-score: {f1_score}")
    print("============================================")
    pass


def main():
    total_results: Dict[DataSelector, List[Result]] = dict()
    for data_selector in data_selectors:
        total_results.setdefault(data_selector, run_pipeline(data_selector))

    for data_selector, result in total_results.items():
        print_results(data_selector, result)

    if len(data_selectors) < 2:
        print("Not enough selectors for a combined f1 score")
        return

    all_combinations = combine_results(total_results)

    for combination in all_combinations:
        f1_score = 0
        for selector, result in combination:
            f1_score += result.f1
        f1_score /= len(combination)

        print_combination(combination, f1_score)


main()
