from typing import List, Dict

from algorithms.Type1 import Type1
from loading.LoadingUtils import LoadingUtils
from utils.Algorithm import Algorithm
from utils.Result import Result

data = LoadingUtils.read_data(filename=r'./data_laptop_absa.json')

algorithms: List[Algorithm] = [Type1()]

results: Dict[str, Result] = dict()
for algorithm in algorithms:
    result = algorithm.execute(data)
    results.setdefault(algorithm.get_name(), result)

for key, value in results.items():
    print("============================================")
    print('Name: ' + str(key))
    print('Precision: ' + str(value.precision))
    print('Recall: ' + str(value.recall))
    print('F1-measure: ' + str(value.f1))
    print("============================================")
    print("")

