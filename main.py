from typing import List

from algorithms.Type1 import Type1
from loading.LoadingUtils import LoadingUtils
from utils.Algorithm import Algorithm
from utils.Result import Result

data = LoadingUtils.readData(filename=r'./data_laptop_absa.json')

algorithms: List[Algorithm] = [Type1()]

results: List[Result] = []
for algorithm in algorithms:
    result = algorithm.execute(data)
    results.append(result)

for result in results:
    print('Precision: ' + str(result.precision))
    print('Recall: ' + str(result.recall))
    print('F1-measure: ' + str(result.f1))
