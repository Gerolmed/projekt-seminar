import copy
from typing import List

from loading.LoadingUtils import LoadedData
from utils.Data import BasicData
from utils.DataProvider import DataProvider
from utils.DataSelector import DataSelector


class ComparisonDataPreparation(DataProvider):
    """Prepares the data for the preparation classifier. Check out its docs for more information"""
    def execute(self, rawData: LoadedData, test_ids: List[str], data_selector: DataSelector) -> BasicData:
        rawData = copy.deepcopy(rawData)

        keys = list(rawData.keys())

        keys_train = list(filter(lambda key: key not in test_ids, keys))
        keys_test = list(filter(lambda key: key in test_ids, keys))

        train_tokens = [rawData[k]['tokens'] for k in keys_train]
        train_labels = list()
        train_labels_uncertainty = list()
        for k in keys_train:
            curr_users = [s for s in rawData[k].keys() if s != 'tokens']
            # for illlustration only the annotation of one user is used here -> curr_users[0]
            train_labels.append(rawData[k][curr_users[0]][data_selector.type_name])
            train_labels_uncertainty.append(rawData[k][curr_users[0]][data_selector.type_name + "_uncertainty"])
        test_tokens = [rawData[k]['tokens'] for k in keys_test]
        test_labels = list()
        test_labels_uncertainty = list()
        for k in keys_test:
            curr_users = [s for s in rawData[k].keys() if s != 'tokens']
            # for illustration only the annotation of one user is used here -> curr_users[0]
            test_labels.append(rawData[k][curr_users[0]][data_selector.type_name])
            test_labels_uncertainty.append(rawData[k][curr_users[0]][data_selector.type_name + "_uncertainty"])

        all_labelclasses = set()
        for ds in [train_labels, test_labels]:
            for row in ds:
                all_labelclasses.update(row)
        all_labelclasses = list(all_labelclasses)
        all_labelclasses.sort()

        labelclass_to_id = dict(zip(all_labelclasses, list(range(len(all_labelclasses)))))

        n_tags = len(list(labelclass_to_id.keys()))
        return BasicData(train_tokens, train_labels, test_tokens, test_labels, n_tags, labelclass_to_id)
